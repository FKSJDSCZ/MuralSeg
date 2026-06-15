# SSMA / HSSMA - (CC-)ELAR decoder head for MMSegmentation (SegFormer-style backbones).
#
# This file implements the decoder proposed in the accompanying design document:
#   - Align + Project: unify multi-stage encoder features to stage-wise channels
#   - PRP: Progressive Residual Propagation to build per-stage queries
#   - SSMA: Scale-Selective Mix-Attention (softmax / avg scale mixing)
#   - HSSMA: Head-wise Scale-Selective Mix-Attention (+ optional diversity regularization)
#   - Multi-stage head: upsample decoded features and fuse for segmentation logits
#   - ELAR: Edge-aware Local Affinity Refinement on logits (optional)
#   - CC-ELAR: Class-Conditional ELAR (optional)
#
# Notes on stage indexing:
#   - stage 0: highest resolution (e.g., H/4)
#   - stage N-1: lowest resolution (e.g., H/32)
#
# `source_indices` convention:
#   - `source_indices[t]` is a list of integers specifying which sources are used
#     when decoding stage `t`.
#   - Non-negative integers refer to encoder stages after Align+Project, i.e.
#       0..N-1 correspond to {Z_0, Z_1, ..., Z_{N-1}}
#   - Use -1 to include the previous decoded feature (Up(D_{t+1})) as a source.
#     (-1 is invalid for the coarsest stage N-1 because there is no previous decode.)

import math
from typing import List, Optional, Sequence, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize


def _to_list(x: Any, length: int):
    """Broadcast scalar to list, keep list/tuple as-is."""
    if isinstance(x, (list, tuple)):
        if len(x) != length:
            raise ValueError(f'Expected length {length}, but got {len(x)}')
        return list(x)
    return [x for _ in range(length)]


def _unique_preserve_order(items: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(int(x))
    return out


class ChannelLayerNorm(BaseModule):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x


class MixFFN2D(BaseModule):
    """A 2D version of SegFormer's MixFFN: 1x1 -> DWConv3x3 -> GELU -> 1x1."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=feedforward_channels,
            bias=True,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(ffn_drop)
        self.fc2 = nn.Conv2d(feedforward_channels, embed_dims, kernel_size=1, bias=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _window_partition(
    x: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, C, H, W)
        window_size: window size

    Returns:
        windows: (B*nW, ws*ws, C)
        padded_hw: (Hp, Wp)
        pad_hw: (pad_h, pad_w)
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = x.shape[2], x.shape[3]

    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, ws, ws, C)
    windows = x.view(-1, window_size * window_size, C)
    return windows, (Hp, Wp), (pad_h, pad_w)


def _window_reverse(
    windows: torch.Tensor,
    window_size: int,
    padded_hw: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    """Reverse windows back to feature map."""
    Hp, Wp = padded_hw
    C = windows.shape[-1]
    num_h = Hp // window_size
    num_w = Wp // window_size

    x = windows.view(batch_size, num_h, num_w, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, C, Hp, Wp)
    return x


class WindowCrossAttention(BaseModule):
    """Window-based cross-attention on NCHW tensors (q,k,v share same spatial shape)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos_bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} must be divisible by num_heads {num_heads}')

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rel_pos_bias = bool(use_rel_pos_bias)

        if self.use_rel_pos_bias:
            ws = self.window_size
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads),
            )

            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, ws, ws)
            coords_flatten = torch.flatten(coords, 1)  # (2, L)

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, L, L)
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (L, L, 2)
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 0] *= 2 * ws - 1
            relative_position_index = relative_coords.sum(-1)  # (L, L)

            self.register_buffer('relative_position_index', relative_position_index, persistent=False)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, C, H, W)
        B, C, H, W = q.shape
        ws = self.window_size

        q_w, (Hp, Wp), (pad_h, pad_w) = _window_partition(q, ws)
        k_w, _, _ = _window_partition(k, ws)
        v_w, _, _ = _window_partition(v, ws)

        BnW, L, _ = q_w.shape

        # (BnW, L, C) -> (BnW, heads, L, head_dim)
        qh = q_w.view(BnW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kh = k_w.view(BnW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vh = v_w.view(BnW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qh = qh * self.scale
        attn = qh @ kh.transpose(-2, -1)  # (BnW, heads, L, L)

        if self.use_rel_pos_bias:
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            bias = bias.view(L, L, self.num_heads).permute(2, 0, 1).contiguous()  # (heads, L, L)
            attn = attn + bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ vh  # (BnW, heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(BnW, L, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = _window_reverse(out, ws, (Hp, Wp), B)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()
        return out


class GlobalCrossAttention(BaseModule):
    """Global cross-attention with optional spatial reduction on K/V.

    This follows SegFormer's efficient attention idea: reduce K/V spatial size
    by strided convolution when sr_ratio > 1.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} must be divisible by num_heads {num_heads}')

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.sr_ratio = int(sr_ratio)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=True)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.sr_norm = None

        # NOTE(A2): We intentionally implement attention explicitly (no nn.MultiheadAttention)
        # to avoid the implicit q/k/v projections inside MHA. Q/K/V are already projected
        # in the caller (SSMA/HSSMA blocks), matching the style of U-MixFormer cross-attn.
        # IMPORTANT: keep the order "sr -> (k,v)" unchanged.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, C, H, W)
        B, C, H, W = q.shape

        q_seq = q.flatten(2).transpose(1, 2)  # (B, Nq, C)

        if self.sr is not None:
            k = self.sr(k)
            v = self.sr(v)
            k_seq = k.flatten(2).transpose(1, 2)
            v_seq = v.flatten(2).transpose(1, 2)
            k_seq = self.sr_norm(k_seq)
            v_seq = self.sr_norm(v_seq)
        else:
            k_seq = k.flatten(2).transpose(1, 2)
            v_seq = v.flatten(2).transpose(1, 2)

        # (B, N, C) -> (B, heads, N, head_dim)
        Nq = q_seq.shape[1]
        Nk = k_seq.shape[1]
        qh = q_seq.view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kh = k_seq.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        vh = v_seq.view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        qh = qh * self.scale
        attn = qh @ kh.transpose(-2, -1)  # (B, heads, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ vh  # (B, heads, Nq, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


class StripCrossAttention(BaseModule):
    """Strip / low-rank cross-attention.

    Motivation (Ablation-B):
      - Standard attention uses Q,K of dimension head_dim, resulting in O(Nq*Nk*head_dim)
        QK^T cost per head.
      - Strip attention compresses **Q and K** to a small dimension `qk_dim` per head
        (often 1), while keeping V at full head_dim:
            Q̄ = W_q(Q) ∈ R^{Nq × qk_dim},  K̄ = W_k(K) ∈ R^{Nk × qk_dim}
        This reduces QK^T cost to O(Nq*Nk*qk_dim) with minimal accuracy drop.

    Inputs are NCHW tensors. K/V can be spatially reduced via `sr_ratio` (SegFormer-style).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        qk_dim: int = 1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} must be divisible by num_heads {num_heads}')

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads

        self.qk_dim = int(qk_dim)
        if self.qk_dim <= 0:
            raise ValueError('qk_dim must be a positive integer')

        self.scale = self.qk_dim ** -0.5
        self.sr_ratio = int(sr_ratio)

        if self.sr_ratio > 1:
            # IMPORTANT: keep the order "sr -> (k,v)" unchanged (same convention as GlobalCrossAttention).
            self.sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=True)
            self.sr_norm = nn.LayerNorm(self.dim)
        else:
            self.sr = None
            self.sr_norm = None

        # Reduce Q/K to (heads*qk_dim) for computing attention weights
        self.q_reduce = nn.Linear(self.dim, self.num_heads * self.qk_dim, bias=True)
        self.k_reduce = nn.Linear(self.dim, self.num_heads * self.qk_dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, C, H, W)
        B, C, H, W = q.shape

        q_seq = q.flatten(2).transpose(1, 2).contiguous()  # (B, Nq, C)
        q_bar = self.q_reduce(q_seq)  # (B, Nq, heads*qk_dim)
        q_bar = q_bar.view(B, -1, self.num_heads, self.qk_dim).permute(0, 2, 1, 3).contiguous()  # (B,heads,Nq,qk_dim)

        if self.sr is not None:
            k = self.sr(k)
            v = self.sr(v)

        k_seq = k.flatten(2).transpose(1, 2).contiguous()  # (B, Nk, C)
        v_seq = v.flatten(2).transpose(1, 2).contiguous()  # (B, Nk, C)

        if self.sr_norm is not None:
            k_seq = self.sr_norm(k_seq)
            v_seq = self.sr_norm(v_seq)

        k_bar = self.k_reduce(k_seq).view(B, -1, self.num_heads, self.qk_dim).permute(
            0, 2, 1, 3,
        ).contiguous()  # (B,heads,Nk,qk_dim)
        v_h = v_seq.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()  # (B,heads,Nk,head_dim)

        attn = (q_bar * self.scale) @ k_bar.transpose(-2, -1)  # (B,heads,Nq,Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v_h  # (B,heads,Nq,head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, C)  # (B,Nq,C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


class SSMABlock(BaseModule):
    """Scale-Selective Mix-Attention block (cross-attn + FFN, no self-attn).

    - Input query `q` is assumed already normalized (PRP does LN outside).
    - Candidate sources are aligned to the same spatial size as `q`, and have
      the same channel dimension `dim` (because SSMA mixes sources per-pixel).

    Supported scale-mix modes:
      - softmax: per-pixel softmax over sources (default)
      - avg: uniform average over sources

    Additional ablations:
      - (B) Strip / low-rank cross-attention: `use_strip_attn=True`
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        gate_channels: int,
        mlp_ratio: int,
        mix_mode: str = 'softmax',  # softmax | avg
        use_window_attn: bool = False,
        use_rel_pos_bias: bool = True,
        sr_ratio: int = 1,
        # Ablation-B: strip attention
        use_strip_attn: bool = False,
        strip_qk_dim: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        mix_mode = str(mix_mode).lower()
        if mix_mode not in {'softmax', 'avg'}:
            raise ValueError(f'Unsupported mix_mode="{mix_mode}". Only softmax|avg are supported.')

        self.dim = int(dim)
        self.mix_mode = mix_mode

        # Scale-selection projection W_a
        gate_channels = int(min(gate_channels, dim))
        self.gate_channels = gate_channels
        self.gate_proj = nn.Conv2d(dim, gate_channels, kernel_size=1, bias=False)

        # Q/K/V projections
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.use_strip_attn = bool(use_strip_attn)

        if self.use_strip_attn:
            self.attn = StripCrossAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
                qk_dim=int(strip_qk_dim),
            )
        elif use_window_attn:
            self.attn = WindowCrossAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_rel_pos_bias=use_rel_pos_bias,
            )
        else:
            self.attn = GlobalCrossAttention(
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.post_attn_norm = ChannelLayerNorm(dim)
        self.ffn = MixFFN2D(dim, feedforward_channels=int(mlp_ratio) * dim, ffn_drop=ffn_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _mix_kv(self, q: torch.Tensor, sources: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate K/V from multiple sources using the configured mix mode."""
        if len(sources) == 0:
            raise ValueError('SSMA requires at least one source feature.')

        # NOTE(A1): to reduce redundant 1x1 projections, we first mix features (Z)
        # and then generate K/V once. This is mathematically equivalent to
        # "mix after projection" because Conv1x1 is linear and
        # \sum_s alpha_s = 1 per pixel (softmax / avg).

        S = len(sources)

        if self.mix_mode == 'avg':
            z_stack = torch.stack(sources, dim=1)  # (B,S,C,H,W)
            z_mix = z_stack.mean(dim=1)
            k = self.k_proj(z_mix)
            v = self.v_proj(z_mix)
            return k, v

        # softmax per-pixel scale selection
        q_gate = self.gate_proj(q)  # (B, d_a, H, W)
        scores = []
        for z in sources:
            z_gate = self.gate_proj(z)
            s = (q_gate * z_gate).sum(dim=1, keepdim=True) / math.sqrt(self.gate_channels)  # (B,1,H,W)
            scores.append(s)
        scores = torch.cat(scores, dim=1)  # (B,S,H,W)
        alpha = scores.softmax(dim=1).unsqueeze(2)  # (B,S,1,H,W)

        z_stack = torch.stack(sources, dim=1)  # (B,S,C,H,W)
        z_mix = (z_stack * alpha).sum(dim=1)
        k = self.k_proj(z_mix)
        v = self.v_proj(z_mix)
        return k, v

    def forward(self, q: torch.Tensor, sources: List[torch.Tensor]) -> torch.Tensor:
        # q: (B,C,H,W), sources: list[(B,C,H,W)] aligned
        q_attn = self.q_proj(q)
        k, v = self._mix_kv(q, sources)

        o = self.attn(q_attn, k, v)  # (B,C,H,W)
        x = self.post_attn_norm(q + self.drop_path(o))

        x = x + self.drop_path(self.ffn(x))
        return x


class HSSMABlock(BaseModule):
    """Head-wise Scale-Selective Mix-Attention block.

    Compared to :class:`SSMABlock`, HSSMA performs *head-wise* scale selection:
    each attention head uses its own softmax weights over candidate sources.

    This enables different heads to specialize on different scale cues.
    Optionally, a diversity regularization (L_div) can be computed from the
    head-wise mixing weights.

    Notes:
      - For HSSMA, we mix *projected* K/V in head space, i.e. we compute K_s/V_s
        for each source, then mix them using alpha^{(h)}.
      - This keeps the downstream attention code unchanged: the mixed K/V are
        concatenated back to (B,C,H,W) with the standard head channel order.

    Additional ablations:
      - (B) Strip / low-rank cross-attention: `use_strip_attn=True`
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        gate_channels: int,
        mlp_ratio: int,
        mix_mode: str = 'softmax',
        use_window_attn: bool = False,
        use_rel_pos_bias: bool = True,
        sr_ratio: int = 1,
        # Ablation-B: strip attention
        use_strip_attn: bool = False,
        strip_qk_dim: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
        # L_div
        compute_div_loss: bool = False,
        div_loss_max_samples: int = 65536,
    ):
        super().__init__()

        mix_mode = str(mix_mode).lower()
        if mix_mode not in {'softmax', 'avg'}:
            raise ValueError(f'Unsupported mix_mode="{mix_mode}". Only softmax|avg are supported.')

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f'dim {self.dim} must be divisible by num_heads {self.num_heads}')
        self.head_dim = self.dim // self.num_heads
        self.mix_mode = mix_mode

        # Head-wise scale-selection projection.
        # We output `gate_channels_total` channels and split evenly across heads.
        # We make `gate_channels_total` divisible by num_heads so that each head
        # receives the same gate channel budget.
        gate_channels_total = int(min(gate_channels, self.dim))
        gate_channels_total = (gate_channels_total // self.num_heads) * self.num_heads
        gate_channels_total = max(gate_channels_total, self.num_heads)
        self.gate_channels_total = gate_channels_total
        self.gate_channels_per_head = gate_channels_total // self.num_heads
        self.gate_proj = nn.Conv2d(self.dim, gate_channels_total, kernel_size=1, bias=False)

        # Q/K/V projections (same as SSMA)
        self.q_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)

        self.use_strip_attn = bool(use_strip_attn)

        if self.use_strip_attn:
            self.attn = StripCrossAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
                qk_dim=int(strip_qk_dim),
            )
        elif use_window_attn:
            self.attn = WindowCrossAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=int(window_size),
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_rel_pos_bias=use_rel_pos_bias,
            )
        else:
            self.attn = GlobalCrossAttention(
                dim=self.dim,
                num_heads=self.num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sr_ratio=sr_ratio,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.post_attn_norm = ChannelLayerNorm(self.dim)
        self.ffn = MixFFN2D(self.dim, feedforward_channels=int(mlp_ratio) * self.dim, ffn_drop=ffn_drop)

        # L_div options
        self.compute_div_loss = bool(compute_div_loss)
        self.div_loss_max_samples = int(div_loss_max_samples)
        self.last_div_loss: Optional[torch.Tensor] = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _sample_tokens(self, a: torch.Tensor) -> torch.Tensor:
        """Optionally subsample tokens for the diversity loss.

        Args:
            a: (N, heads, S)
        """
        if self.div_loss_max_samples <= 0:
            return a
        N = a.shape[0]
        if N <= self.div_loss_max_samples:
            return a
        # Random sampling (training only). Note: use device RNG.
        idx = torch.randperm(N, device=a.device)[: self.div_loss_max_samples]
        return a.index_select(0, idx)

    def _compute_div_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute L_div from head-wise mixing weights.

        Args:
            alpha: (B, heads, S, H, W)
        """
        B, Hh, S, H, W = alpha.shape
        # (B,heads,S,H,W) -> (N, heads, S)
        a = alpha.permute(0, 3, 4, 1, 2).reshape(-1, Hh, S)
        a = self._sample_tokens(a)
        gram = torch.bmm(a, a.transpose(1, 2))  # (N, heads, heads)
        eye = torch.eye(Hh, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        return (gram - eye).pow(2).mean()

    def _mix_kv(self, q: torch.Tensor, sources: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(sources) == 0:
            raise ValueError('HSSMA requires at least one source feature.')

        B, C, H, W = q.shape
        S = len(sources)

        # Project all sources to K/V once (then mix per-head).
        k_list = [self.k_proj(z) for z in sources]
        v_list = [self.v_proj(z) for z in sources]
        k_stack = torch.stack(k_list, dim=1)  # (B,S,C,H,W)
        v_stack = torch.stack(v_list, dim=1)
        k_stack = k_stack.view(B, S, self.num_heads, self.head_dim, H, W)
        v_stack = v_stack.view(B, S, self.num_heads, self.head_dim, H, W)

        if self.mix_mode == 'avg':
            alpha = torch.full(
                (B, self.num_heads, S, H, W),
                1.0 / float(S),
                device=q.device,
                dtype=q.dtype,
            )
        else:
            # Head-wise per-pixel scale selection
            q_gate = self.gate_proj(q).view(B, self.num_heads, self.gate_channels_per_head, H, W)
            scores = []
            for z in sources:
                z_gate = self.gate_proj(z).view(B, self.num_heads, self.gate_channels_per_head, H, W)
                s = (q_gate * z_gate).sum(dim=2) / math.sqrt(self.gate_channels_per_head)  # (B,heads,H,W)
                scores.append(s.unsqueeze(2))
            scores = torch.cat(scores, dim=2)  # (B,heads,S,H,W)
            alpha = scores.softmax(dim=2)

        # (B,heads,S,H,W) -> (B,S,heads,1,H,W) for broadcast over head_dim
        a = alpha.permute(0, 2, 1, 3, 4).unsqueeze(3)
        k_mix = (k_stack * a).sum(dim=1)  # (B,heads,head_dim,H,W)
        v_mix = (v_stack * a).sum(dim=1)
        k_mix = k_mix.reshape(B, C, H, W)
        v_mix = v_mix.reshape(B, C, H, W)

        # Optional diversity loss
        if self.compute_div_loss and self.mix_mode == 'softmax' and self.training:
            self.last_div_loss = self._compute_div_loss(alpha)
        else:
            self.last_div_loss = None

        return k_mix, v_mix

    def forward(self, q: torch.Tensor, sources: List[torch.Tensor]) -> torch.Tensor:
        q_attn = self.q_proj(q)
        k, v = self._mix_kv(q, sources)
        o = self.attn(q_attn, k, v)

        x = self.post_attn_norm(q + self.drop_path(o))

        x = x + self.drop_path(self.ffn(x))
        return x


class ELARRefine(BaseModule):
    """Edge-aware Local Affinity Refinement (ELAR) on logits.

    It uses a local kxk window affinity (softmax) computed from guidance features
    to propagate/refine logits.

    Implementation uses `F.unfold`, which can be memory-heavy for large H/W.
    """

    def __init__(
        self,
        num_logits_channels: int,
        feat_channels: int,
        kernel_size: int = 5,
        guidance_channels: int = 64,
        theta_channels: int = 16,
        use_prob: bool = True,
        prob_channels: int = 16,
        detach_prob: bool = True,
        num_iters: int = 1,
        residual: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')

        self.num_logits_channels = int(num_logits_channels)
        self.kernel_size = int(kernel_size)
        self.theta_channels = int(theta_channels)
        self.use_prob = bool(use_prob)
        self.detach_prob = bool(detach_prob)
        self.num_iters = int(num_iters)
        self.residual = bool(residual)
        self.residual_weight = float(residual_weight)

        in_g = feat_channels * 2
        if self.use_prob:
            self.prob_proj = nn.Conv2d(num_logits_channels, prob_channels, kernel_size=1, bias=True)
            in_g = in_g + prob_channels
        else:
            self.prob_proj = None

        self.guidance_fuse = nn.Conv2d(in_g, guidance_channels, kernel_size=1, bias=True)
        self.guidance_norm = ChannelLayerNorm(guidance_channels)

        self.theta = nn.Conv2d(guidance_channels, theta_channels, kernel_size=1, bias=True)

        # learnable relative bias (kxk)
        self.rel_bias = nn.Parameter(torch.zeros(kernel_size * kernel_size))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _prob(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_logits_channels == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def forward(self, logits: torch.Tensor, feat_high: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        """Refine logits.

        Args:
            logits: (B, C_out, H, W)
            feat_high: decoded feature at highest resolution (B, d, H, W)
            feat_low: aligned low-level encoder feature at same resolution (B, d, H, W)
        """
        if logits.dim() != 4:
            raise ValueError('logits must be a 4D tensor (B,C,H,W)')

        B, C_out, H, W = logits.shape
        if C_out != self.num_logits_channels:
            raise ValueError(f'logits channels {C_out} != configured {self.num_logits_channels}')

        k = self.kernel_size
        pad = k // 2

        x = logits
        for _ in range(max(1, self.num_iters)):
            # guidance: G = phi_g([D1; Z11; psi(P)])
            g_list = [feat_high, feat_low]
            if self.use_prob:
                p = self._prob(x)
                if self.detach_prob:
                    p = p.detach()
                p = self.prob_proj(p)
                g_list.append(p)

            g = torch.cat(g_list, dim=1)
            g = self.guidance_norm(self.guidance_fuse(g))
            t = self.theta(g)  # (B, d_g, H, W)

            # unfold theta and logits
            t_unf = F.unfold(t, kernel_size=k, padding=pad)  # (B, d_g*k*k, H*W)
            t_unf = t_unf.view(B, self.theta_channels, k * k, H * W).permute(0, 3, 2, 1).contiguous()
            t_center = t.flatten(2).transpose(1, 2).contiguous()  # (B, H*W, d_g)

            # affinity weights: (B, H*W, k*k)
            scores = (t_unf * t_center.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.theta_channels)
            scores = scores + self.rel_bias.view(1, 1, k * k).to(dtype=scores.dtype)
            weights = scores.softmax(dim=-1)

            # propagate logits
            l_unf = F.unfold(x, kernel_size=k, padding=pad)  # (B, C_out*k*k, H*W)
            l_unf = l_unf.view(B, C_out, k * k, H * W).permute(0, 3, 2, 1).contiguous()  # (B,H*W,k*k,C)
            x_ref = (weights.unsqueeze(-1) * l_unf).sum(dim=2)  # (B, H*W, C_out)
            x_ref = x_ref.transpose(1, 2).reshape(B, C_out, H, W).contiguous()

            if self.residual:
                x = x + self.residual_weight * (x_ref - x)
            else:
                x = x_ref

        return x


class CCELARRefine(BaseModule):
    """Class-Conditional Edge-aware Local Affinity Refinement (CC-ELAR).

    This extends :class:`ELARRefine` by adding a *class-consistency* term in the
    affinity computation. Intuitively, besides structural similarity (theta),
    neighbor logits should be aggregated more strongly when their predicted
    class distributions are compatible.

    Affinity score (per pixel p and neighbor q):
        s_{p,q} = <theta_p, theta_q>/sqrt(d_theta)
                + cc_lambda * <phi_p, phi_q>/sqrt(d_phi)
                + b_{p-q}

    where phi is an embedding of class-probabilities (softmax/sigmoid).
    """

    def __init__(
        self,
        num_logits_channels: int,
        feat_channels: int,
        kernel_size: int = 5,
        guidance_channels: int = 64,
        theta_channels: int = 16,
        # class-conditional term
        cc_lambda: float = 1.0,
        # probability embedding
        prob_channels: int = 16,
        detach_prob: bool = True,
        num_iters: int = 1,
        residual: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')

        self.num_logits_channels = int(num_logits_channels)
        self.kernel_size = int(kernel_size)
        self.theta_channels = int(theta_channels)
        self.cc_lambda = float(cc_lambda)
        self.prob_channels = int(prob_channels)
        self.detach_prob = bool(detach_prob)
        self.num_iters = int(num_iters)
        self.residual = bool(residual)
        self.residual_weight = float(residual_weight)

        # In CC-ELAR we always use prob embedding (phi) for the class term.
        # We also concatenate it into guidance to let theta see semantic priors.
        self.prob_proj = nn.Conv2d(self.num_logits_channels, self.prob_channels, kernel_size=1, bias=True)
        in_g = feat_channels * 2 + self.prob_channels

        self.guidance_fuse = nn.Conv2d(in_g, guidance_channels, kernel_size=1, bias=True)
        self.guidance_norm = ChannelLayerNorm(guidance_channels)
        self.theta = nn.Conv2d(guidance_channels, self.theta_channels, kernel_size=1, bias=True)

        # learnable relative bias (kxk)
        self.rel_bias = nn.Parameter(torch.zeros(self.kernel_size * self.kernel_size))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _prob(self, logits: torch.Tensor) -> torch.Tensor:
        if self.num_logits_channels == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def forward(self, logits: torch.Tensor, feat_high: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4:
            raise ValueError('logits must be a 4D tensor (B,C,H,W)')

        B, C_out, H, W = logits.shape
        if C_out != self.num_logits_channels:
            raise ValueError(f'logits channels {C_out} != configured {self.num_logits_channels}')

        k = self.kernel_size
        pad = k // 2

        x = logits
        for _ in range(max(1, self.num_iters)):
            # class embedding phi(P)
            p = self._prob(x)
            if self.detach_prob:
                p = p.detach()
            phi = self.prob_proj(p)  # (B, d_phi, H, W)

            # guidance: G = phi_g([D1; Z11; phi(P)])
            g = torch.cat([feat_high, feat_low, phi], dim=1)
            g = self.guidance_norm(self.guidance_fuse(g))
            t = self.theta(g)  # (B, d_theta, H, W)

            # unfold theta (and phi)
            t_unf = F.unfold(t, kernel_size=k, padding=pad)
            t_unf = t_unf.view(B, self.theta_channels, k * k, H * W).permute(0, 3, 2, 1).contiguous()
            t_center = t.flatten(2).transpose(1, 2).contiguous()

            scores = (t_unf * t_center.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.theta_channels)

            if self.cc_lambda != 0.0:
                phi_unf = F.unfold(phi, kernel_size=k, padding=pad)
                phi_unf = phi_unf.view(B, self.prob_channels, k * k, H * W).permute(0, 3, 2, 1).contiguous()
                phi_center = phi.flatten(2).transpose(1, 2).contiguous()
                scores_phi = (phi_unf * phi_center.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.prob_channels)
                scores = scores + self.cc_lambda * scores_phi

            scores = scores + self.rel_bias.view(1, 1, k * k).to(dtype=scores.dtype)
            weights = scores.softmax(dim=-1)

            # propagate logits
            l_unf = F.unfold(x, kernel_size=k, padding=pad)
            l_unf = l_unf.view(B, C_out, k * k, H * W).permute(0, 3, 2, 1).contiguous()
            x_ref = (weights.unsqueeze(-1) * l_unf).sum(dim=2)
            x_ref = x_ref.transpose(1, 2).reshape(B, C_out, H, W).contiguous()

            if self.residual:
                x = x + self.residual_weight * (x_ref - x)
            else:
                x = x_ref

        return x


class StageELARRefine(BaseModule):
    """Stage-wise ELAR: refine *features* (not logits) with edge-aware local affinity.

    - value: feature map x (B, C, H, W)
    - guidance: concat([x, feat_low, optional prob_embed])
    - affinity: local kxk, computed from theta(guidance)
    - aggregation: shift-based neighbor gathering (memory friendly)
    """

    def __init__(
        self,
        in_channels: int,
        low_channels: int,
        num_classes: int,
        kernel_size: int = 5,
        guidance_channels: int = 64,
        theta_channels: int = 16,
        value_channels: int = 32,
        # optional semantic prior (aux prob)
        use_prob: bool = False,
        prob_channels: int = 16,
        detach_prob: bool = True,
        # refinement
        num_iters: int = 1,
        residual: bool = True,
        residual_weight: float = 1.0,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')

        self.in_channels = int(in_channels)
        self.low_channels = int(low_channels)
        self.num_classes = int(num_classes)
        self.kernel_size = int(kernel_size)
        self.guidance_channels = int(guidance_channels)
        self.theta_channels = int(theta_channels)
        self.value_channels = int(value_channels)

        self.use_prob = bool(use_prob)
        self.prob_channels = int(prob_channels)
        self.detach_prob = bool(detach_prob)

        self.num_iters = int(num_iters)
        self.residual = bool(residual)
        self.residual_weight = float(residual_weight)

        # value projection to reduce cost (optional)
        self.value_proj = (
            nn.Identity()
            if self.value_channels == self.in_channels
            else nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=True)
        )
        self.out_proj = (
            nn.Identity()
            if self.value_channels == self.in_channels
            else nn.Conv2d(self.value_channels, self.in_channels, kernel_size=1, bias=True)
        )

        # optional aux classifier (only to provide prob embedding; can also be supervised)
        if self.use_prob:
            self.aux_pred = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1, bias=True)
            self.prob_proj = nn.Conv2d(self.num_classes, self.prob_channels, kernel_size=1, bias=True)
            in_g = self.in_channels + self.low_channels + self.prob_channels
        else:
            self.aux_pred = None
            self.prob_proj = None
            in_g = self.in_channels + self.low_channels

        self.guidance_fuse = nn.Conv2d(in_g, self.guidance_channels, kernel_size=1, bias=True)
        self.guidance_norm = ChannelLayerNorm(self.guidance_channels)
        self.theta = nn.Conv2d(self.guidance_channels, self.theta_channels, kernel_size=1, bias=True)

        # learnable relative bias (kxk)
        self.rel_bias = nn.Parameter(torch.zeros(self.kernel_size * self.kernel_size))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        self.scale = self.theta_channels ** -0.5

        # for optional deep supervision
        self.last_aux_logits: Optional[torch.Tensor] = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def _prob(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return logits.sigmoid()
        return logits.softmax(dim=1)

    def forward(self, x: torch.Tensor, feat_low: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('x must be a 4D tensor (B,C,H,W)')
        if feat_low.dim() != 4:
            raise ValueError('feat_low must be a 4D tensor (B,C,H,W)')
        if feat_low.shape[2:] != x.shape[2:]:
            raise ValueError('feat_low must be aligned to x spatial size before calling StageELARRefine')

        B, C, H, W = x.shape
        k = self.kernel_size
        pad = k // 2

        y = x
        self.last_aux_logits = None

        for _ in range(max(1, self.num_iters)):
            # --- build guidance and theta ---
            g_list = [y, feat_low]

            if self.use_prob:
                assert self.aux_pred is not None and self.prob_proj is not None
                aux_logits = self.aux_pred(y)  # (B, num_classes, H, W)
                self.last_aux_logits = aux_logits
                p = self._prob(aux_logits)
                if self.detach_prob:
                    p = p.detach()
                p = self.prob_proj(p)  # (B, prob_channels, H, W)
                g_list.append(p)

            g = torch.cat(g_list, dim=1)
            g = self.guidance_norm(self.guidance_fuse(g))
            t = self.theta(g)  # (B, d_theta, H, W)

            # --- compute local affinity weights (shift-based) ---
            t_pad = F.pad(t, (pad, pad, pad, pad))
            v = self.value_proj(y)
            v_pad = F.pad(v, (pad, pad, pad, pad))

            scores = []
            neigh_vs = []
            # offsets order consistent with rel_bias indexing
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    t_nb = t_pad[:, :, pad + dy: pad + dy + H, pad + dx: pad + dx + W]
                    s = (t * t_nb).sum(dim=1) * self.scale  # (B, H, W)
                    scores.append(s)

                    v_nb = v_pad[:, :, pad + dy: pad + dy + H, pad + dx: pad + dx + W]
                    neigh_vs.append(v_nb)

            score = torch.stack(scores, dim=1)  # (B, K, H, W)
            score = score + self.rel_bias.view(1, -1, 1, 1)
            w = score.softmax(dim=1)  # (B, K, H, W)

            # --- aggregate values ---
            v_ref = 0.0
            for i, v_nb in enumerate(neigh_vs):
                v_ref = v_ref + w[:, i:i + 1] * v_nb

            y_ref = self.out_proj(v_ref)

            if self.residual:
                y = y + self.residual_weight * (y_ref - y)
            else:
                y = y_ref

        return y


@MODELS.register_module()
class SSMAHeadv2(BaseDecodeHead):
    """SSMA/HSSMA - (CC-)ELAR decoder head.

    This head supports the proposed SSMA + PRP decoding pipeline (and its
    head-wise variant HSSMA) with optional multi-stage fusion and optional
    (CC-)ELAR refinement.

    Key requirements for this refactored version:
      - No dict-wrapped configs in __init__ (explicit args)
      - No decode_mode switch (only SSMA-ELAR forward)
      - SSMA mix_mode supports only: softmax | avg
      - Sources are specified by `source_indices` (2D list), no source_mode

    Args:
        interpolate_mode: upsample mode used by `resize`.
        use_ssma: enable mix-attention blocks (SSMA or HSSMA).
        ssma_type: select 'ssma' or 'hssma'.
        hssma_*: options for HSSMA (e.g., L_div).
        use_prp: enable PRP query construction (residual Up).
        use_multi_stage_fuse: fuse decoded features from all stages.
        use_elar: enable logit refinement.
        elar_type: select 'elar' or 'cc_elar'.

        fuse_type: fusion module type for multi-stage fuse. 'conv1x1' or 'multi_conv'.
        fuse_num_convs: number of conv layers when fuse_type='multi_conv'.
        fuse_kernel_sizes: per-layer kernel sizes for 'multi_conv'.
        fuse_mid_channels: intermediate channels for 'multi_conv'.
        fuse_act_last: whether to apply activation on the last fusion layer.
        fuse_out_channels: output channels of the fused feature (B1).

        source_indices: 2D list specifying sources per stage. Use -1 to include prev-dec.

        downsample_mode: how to align encoder features when downsampling.
            Options: 'avg' | 'bilinear' | 'area'.

        ssma_*: (H)SSMA block hyper-parameters.
        elar_*: (CC-)ELAR hyper-parameters.

        **kwargs: BaseDecodeHead args (in_channels, channels, num_classes, etc.).
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        # Ablation switches
        use_ssma: bool = True,
        ssma_type: str = 'ssma',
        # HSSMA (+L_div)
        hssma_use_div_loss: bool = False,
        hssma_div_loss_weight: float = 0.0,
        hssma_div_loss_max_samples: int = 65536,
        use_prp: bool = True,
        use_multi_stage_fuse: bool = True,
        use_elar: bool = True,
        # Multi-stage fusion
        fuse_type: str = 'conv1x1',
        fuse_num_convs: int = 3,
        fuse_kernel_sizes: Optional[Sequence[int]] = None,
        fuse_mid_channels: Optional[int] = None,
        fuse_act_last: bool = False,
        fuse_out_channels: Optional[int] = None,
        # SSMA: source selection & alignment
        source_indices: Optional[Sequence[Sequence[int]]] = None,
        downsample_mode: str = 'bilinear',
        # SSMA: cross-layer decoder memory for K/V sources (Ablation A)
        use_cross_layer_kv: bool = False,
        cross_layer_kv_depth: int = -1,
        # SSMA: attention + FFN
        ssma_num_heads: Union[int, Sequence[int]] = 8,
        ssma_window_size: Union[int, Sequence[int]] = 7,
        ssma_use_window_attn: Union[bool, Sequence[bool]] = False,
        # Ablation-B: strip / low-rank cross-attention (overrides window/global when True)
        ssma_use_strip_attn: Union[bool, Sequence[bool]] = False,
        ssma_strip_qk_dim: int = 1,
        ssma_use_rel_pos_bias: bool = True,
        ssma_sr_ratio: Union[int, Sequence[int]] = 1,
        ssma_mix_mode: str = 'softmax',
        ssma_gate_channels: int = 64,
        ssma_mlp_ratio: Union[int, Sequence[int]] = 4,
        ssma_attn_drop: float = 0.0,
        ssma_proj_drop: float = 0.0,
        ssma_ffn_drop: float = 0.0,
        ssma_drop_path: Union[float, Sequence[float]] = 0.0,
        # (CC-)ELAR
        elar_type: str = 'elar',
        cc_elar_lambda: float = 1.0,
        elar_kernel_size: int = 5,
        elar_guidance_channels: int = 64,
        elar_theta_channels: int = 16,
        elar_use_prob: bool = True,
        elar_prob_channels: int = 16,
        elar_detach_prob: bool = True,
        elar_num_iters: int = 1,
        elar_residual: bool = True,
        elar_residual_weight: float = 1.0,
        # Stage-ELAR (feature-level, inserted into decoder stages)
        use_stage_elar: bool = False,
        stage_elar_stages: Optional[Sequence[int]] = None,  # e.g. [0,1] or [0,1,2,3]
        stage_elar_low_index: int = 0,  # which proj_feats[k] used as "low" guidance
        stage_elar_kernel_size: Union[int, Sequence[int]] = 5,
        stage_elar_guidance_channels: int = 64,
        stage_elar_theta_channels: Union[int, Sequence[int]] = 16,
        stage_elar_value_channels: Union[int, Sequence[int]] = 32,
        stage_elar_use_prob: Union[bool, Sequence[bool]] = False,
        stage_elar_prob_channels: int = 16,
        stage_elar_detach_prob: bool = True,
        stage_elar_num_iters: int = 1,
        stage_elar_residual: bool = True,
        stage_elar_residual_weight: float = 1.0,
        # optional deep supervision on aux logits inside Stage-ELAR
        stage_elar_aux_loss: bool = False,
        stage_elar_aux_weight: Union[float, Sequence[float]] = 0.2,
        stage_elar_aux_stages: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        # ---- BaseDecodeHead bookkeeping ----
        raw_in_channels = kwargs.get('in_channels', None)
        if isinstance(raw_in_channels, int):
            num_inputs = 1
        else:
            if not isinstance(raw_in_channels, (list, tuple)):
                raise TypeError('`in_channels` must be int or list/tuple of ints.')
            num_inputs = len(raw_in_channels)

        raw_channels = kwargs.get('channels', None)
        if raw_channels is None:
            raise ValueError('`channels` must be provided in decode_head cfg.')

        # We allow `channels` to be int OR stage-wise list/tuple aligned with in_index.
        self.stage_channels = _to_list(raw_channels, num_inputs)

        # B1: allow a wider fused feature than stage0 channels.
        self.fuse_out_channels = int(self.stage_channels[0]) if fuse_out_channels is None else int(fuse_out_channels)

        # BaseDecodeHead expects an int `channels` to build cls_seg.
        kwargs['channels'] = int(self.fuse_out_channels)

        super().__init__(input_transform='multiple_select', **kwargs)

        # ---- Store switches / common settings ----
        self.interpolate_mode = str(interpolate_mode)
        self.use_ssma = bool(use_ssma)
        self.use_prp = bool(use_prp)
        self.use_multi_stage_fuse = bool(use_multi_stage_fuse)
        self.use_elar = bool(use_elar)

        # ---- Ablation A: cross-layer decoder memory for K/V sources ----
        self.use_cross_layer_kv = bool(use_cross_layer_kv) and self.use_ssma
        self.cross_layer_kv_depth = int(cross_layer_kv_depth)
        if self.cross_layer_kv_depth < -1:
            raise ValueError('cross_layer_kv_depth must be -1 (all) or >= 0.')

        # SSMA/HSSMA selection
        self.ssma_type = str(ssma_type).lower()
        if self.ssma_type not in {'ssma', 'hssma'}:
            raise ValueError("ssma_type must be 'ssma' or 'hssma'.")

        # HSSMA (+L_div)
        self.hssma_use_div_loss = bool(hssma_use_div_loss)
        self.hssma_div_loss_weight = float(hssma_div_loss_weight)
        self.hssma_div_loss_max_samples = int(hssma_div_loss_max_samples)
        self._last_div_loss: Optional[torch.Tensor] = None

        # ELAR / CC-ELAR selection
        self.elar_type = str(elar_type).lower().replace('-', '_')
        if self.elar_type not in {'elar', 'cc_elar'}:
            raise ValueError("elar_type must be 'elar' or 'cc_elar'.")
        self.cc_elar_lambda = float(cc_elar_lambda)

        num_stages = len(self.in_channels)
        if num_stages != len(self.in_index):
            raise ValueError('in_channels and in_index must have the same length.')
        if num_stages != len(self.stage_channels):
            raise ValueError('stage_channels must have the same length as in_channels.')

        # ---- Validate / normalize source_indices ----
        self.source_indices: Optional[List[List[int]]] = None
        if self.use_ssma:
            if source_indices is None:
                raise ValueError('When use_ssma=True, you must provide `source_indices` (2D list).')
            self.source_indices = self._validate_and_normalize_source_indices(source_indices, num_stages)

        # Whether we ever need Up(D_{t+1}) (for PRP and/or as a SSMA source)
        need_prev = self.use_prp
        if self.use_ssma and self.source_indices is not None:
            need_prev = need_prev or any((-1 in idxs) for idxs in self.source_indices)

        # (A) Cross-layer decoder memory also needs Up(D_{t+1}) when enabled (depth >= 1).
        if self.use_ssma and self.use_cross_layer_kv and (
                self.cross_layer_kv_depth == -1 or self.cross_layer_kv_depth >= 1):
            need_prev = True

        # ---- Align + Project (encoder stage-wise) ----
        self.proj_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.stage_channels[i],
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                for i in range(num_stages)
            ],
        )

        # ---- PRP query LN (stage-wise) ----
        self.q_norms = nn.ModuleList([ChannelLayerNorm(self.stage_channels[i]) for i in range(num_stages)])

        # ---- Prev feature projection (Up(D_{t+1}) -> stage t channels), only if needed ----
        self.prev_projs: Optional[nn.ModuleList]
        if need_prev:
            prev_projs = nn.ModuleList()
            for t in range(num_stages):
                if t == (num_stages - 1):
                    # coarsest stage has no previous decoded feature
                    prev_projs.append(nn.Identity())
                    continue
                c_in = int(self.stage_channels[t + 1])
                c_out = int(self.stage_channels[t])
                if c_in == c_out:
                    prev_projs.append(nn.Identity())
                else:
                    prev_projs.append(nn.Conv2d(c_in, c_out, kernel_size=1, bias=True))
            self.prev_projs = prev_projs
        else:
            self.prev_projs = None

        # ---- (A) Cross-layer decoder memory projections (only if enabled) ----
        # Project deeper decoded features D_{t+k} (k>=2) -> stage t channels.
        # The immediate previous decoded feature D_{t+1} is handled by `prev_projs`.
        self.dec_projs: Optional[nn.ModuleList] = None
        if self.use_ssma and self.use_cross_layer_kv and (
                self.cross_layer_kv_depth == -1 or self.cross_layer_kv_depth >= 2):
            dec_projs = nn.ModuleList()
            for t in range(num_stages):
                md = nn.ModuleDict()
                for s in range(t + 2, num_stages):
                    hop = s - t
                    if self.cross_layer_kv_depth != -1 and hop > self.cross_layer_kv_depth:
                        break
                    c_in = int(self.stage_channels[s])
                    c_out = int(self.stage_channels[t])
                    if c_in == c_out:
                        md[str(s)] = nn.Identity()
                    else:
                        md[str(s)] = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
                dec_projs.append(md)
            self.dec_projs = dec_projs

        # ---- SSMA blocks + source projections (only if SSMA is enabled) ----
        self.downsample_mode = str(downsample_mode).lower()
        if self.downsample_mode not in {'avg', 'bilinear', 'area'}:
            raise ValueError('downsample_mode must be one of: avg | bilinear | area')

        self.src_projs: Optional[nn.ModuleList] = None
        self.ssma_blocks: Optional[nn.ModuleList] = None

        if self.use_ssma:
            # Per-stage hyper-parameters
            num_heads = _to_list(ssma_num_heads, num_stages)
            window_size = _to_list(ssma_window_size, num_stages)
            use_window_attn = _to_list(ssma_use_window_attn, num_stages)
            use_strip_attn = _to_list(ssma_use_strip_attn, num_stages)
            strip_qk_dim = int(ssma_strip_qk_dim)
            mlp_ratio = _to_list(ssma_mlp_ratio, num_stages)
            sr_ratio = _to_list(ssma_sr_ratio, num_stages)
            drop_path = _to_list(ssma_drop_path, num_stages)

            # Build only the source projection modules that will actually be used.
            # src_projs[t][str(s)] projects encoder stage s (after proj) -> stage t channels.
            src_projs = nn.ModuleList()
            for t in range(num_stages):
                assert self.source_indices is not None
                used_sources = [i for i in self.source_indices[t] if i >= 0]
                used_sources = _unique_preserve_order(used_sources)

                md = nn.ModuleDict()
                for s in used_sources:
                    c_in = int(self.stage_channels[s])
                    c_out = int(self.stage_channels[t])
                    if c_in == c_out:
                        md[str(s)] = nn.Identity()
                    else:
                        md[str(s)] = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
                src_projs.append(md)
            self.src_projs = src_projs

            ssma_blocks = nn.ModuleList()
            for t in range(num_stages):
                c = int(self.stage_channels[t])
                h = int(num_heads[t])
                if c % h != 0:
                    raise ValueError(
                        f'stage_channels[{t}]={c} must be divisible by ssma_num_heads[{t}]={h}',
                    )
                block_kwargs = dict(
                    dim=c,
                    num_heads=h,
                    window_size=int(window_size[t]),
                    gate_channels=int(ssma_gate_channels),
                    mlp_ratio=int(mlp_ratio[t]),
                    mix_mode=str(ssma_mix_mode),
                    use_window_attn=bool(use_window_attn[t]),
                    use_strip_attn=bool(use_strip_attn[t]),
                    strip_qk_dim=int(strip_qk_dim),
                    use_rel_pos_bias=bool(ssma_use_rel_pos_bias),
                    sr_ratio=int(sr_ratio[t]),
                    attn_drop=float(ssma_attn_drop),
                    proj_drop=float(ssma_proj_drop),
                    ffn_drop=float(ssma_ffn_drop),
                    drop_path=float(drop_path[t]),
                )

                if self.ssma_type == 'ssma':
                    ssma_blocks.append(SSMABlock(**block_kwargs))
                else:
                    ssma_blocks.append(
                        HSSMABlock(
                            **block_kwargs,
                            compute_div_loss=self.hssma_use_div_loss,
                            div_loss_max_samples=self.hssma_div_loss_max_samples,
                        ),
                    )
            self.ssma_blocks = ssma_blocks

        # ---- Stage-ELAR refinement (feature-level, inserted per-stage) ----
        self.use_stage_elar = bool(use_stage_elar)
        self.stage_elar_blocks: Optional[nn.ModuleList] = None
        self.stage_elar_stages: List[int] = []
        self.stage_elar_low_index = int(stage_elar_low_index)

        self.stage_elar_aux_loss = bool(stage_elar_aux_loss)
        self.stage_elar_aux_stages: List[int] = []
        self.stage_elar_aux_weight_list: List[float] = []

        self._last_stage_aux_logits: Optional[List[Optional[torch.Tensor]]] = None

        if self.use_stage_elar:
            if self.stage_elar_low_index < 0 or self.stage_elar_low_index >= num_stages:
                raise ValueError('stage_elar_low_index must be in [0, num_stages-1].')

            if stage_elar_stages is None:
                # default: refine high-res stages first (you can change to [0,1,2,3])
                stage_elar_stages = [0, 1]
            self.stage_elar_stages = _unique_preserve_order(stage_elar_stages)
            for s in self.stage_elar_stages:
                if s < 0 or s >= num_stages:
                    raise ValueError(f'stage_elar_stages contains invalid stage index {s}.')

            ks_list = _to_list(stage_elar_kernel_size, num_stages)
            th_list = _to_list(stage_elar_theta_channels, num_stages)
            vc_list = _to_list(stage_elar_value_channels, num_stages)
            up_list = _to_list(stage_elar_use_prob, num_stages)

            # aux loss bookkeeping
            if self.stage_elar_aux_loss:
                if stage_elar_aux_stages is None:
                    stage_elar_aux_stages = list(self.stage_elar_stages)
                self.stage_elar_aux_stages = _unique_preserve_order(stage_elar_aux_stages)
                for s in self.stage_elar_aux_stages:
                    if s not in self.stage_elar_stages:
                        raise ValueError(
                            f'stage_elar_aux_stages contains stage {s} not in stage_elar_stages.',
                        )
                # weights per stage (broadcast scalar)
                self.stage_elar_aux_weight_list = [float(x) for x in _to_list(stage_elar_aux_weight, num_stages)]
            else:
                self.stage_elar_aux_stages = []
                self.stage_elar_aux_weight_list = [0.0 for _ in range(num_stages)]

            low_c = int(self.stage_channels[self.stage_elar_low_index])

            blocks = nn.ModuleList()
            for t in range(num_stages):
                if t in self.stage_elar_stages:
                    blocks.append(
                        StageELARRefine(
                            in_channels=int(self.stage_channels[t]),
                            low_channels=low_c,
                            num_classes=int(self.out_channels),
                            kernel_size=int(ks_list[t]),
                            guidance_channels=int(stage_elar_guidance_channels),
                            theta_channels=int(th_list[t]),
                            value_channels=int(vc_list[t]),
                            use_prob=bool(up_list[t]),
                            prob_channels=int(stage_elar_prob_channels),
                            detach_prob=bool(stage_elar_detach_prob),
                            num_iters=int(stage_elar_num_iters),
                            residual=bool(stage_elar_residual),
                            residual_weight=float(stage_elar_residual_weight),
                        ),
                    )
                else:
                    blocks.append(nn.Identity())

            self.stage_elar_blocks = blocks

        # ---- Multi-stage fusion conv (only if enabled) ----
        self.fusion_conv: Optional[nn.Module] = None
        if self.use_multi_stage_fuse:
            fuse_in_channels = int(sum(self.stage_channels))
            fuse_out_channels = int(self.fuse_out_channels)
            self.fusion_conv = self._build_fusion_module(
                in_channels=fuse_in_channels,
                out_channels=fuse_out_channels,
                fuse_type=fuse_type,
                num_convs=fuse_num_convs,
                kernel_sizes=fuse_kernel_sizes,
                mid_channels=fuse_mid_channels,
                act_last=fuse_act_last,
            )

        # If multi-stage fuse is disabled but cls_seg expects `fuse_out_channels`,
        # project the stage0 feature to the configured width.
        self.out_proj: Optional[nn.Module] = None
        if (not self.use_multi_stage_fuse) and (int(self.stage_channels[0]) != int(self.fuse_out_channels)):
            self.out_proj = ConvModule(
                in_channels=int(self.stage_channels[0]),
                out_channels=int(self.fuse_out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

        # ---- ELAR refinement (only if enabled) ----
        self.elar: Optional[nn.Module] = None
        if self.use_elar:
            feat_ch = int(self.stage_channels[0])
            if self.elar_type == 'elar':
                self.elar = ELARRefine(
                    num_logits_channels=self.out_channels,
                    feat_channels=feat_ch,
                    kernel_size=int(elar_kernel_size),
                    guidance_channels=int(elar_guidance_channels),
                    theta_channels=int(elar_theta_channels),
                    use_prob=bool(elar_use_prob),
                    prob_channels=int(elar_prob_channels),
                    detach_prob=bool(elar_detach_prob),
                    num_iters=int(elar_num_iters),
                    residual=bool(elar_residual),
                    residual_weight=float(elar_residual_weight),
                )
            else:
                self.elar = CCELARRefine(
                    num_logits_channels=self.out_channels,
                    feat_channels=feat_ch,
                    kernel_size=int(elar_kernel_size),
                    guidance_channels=int(elar_guidance_channels),
                    theta_channels=int(elar_theta_channels),
                    cc_lambda=float(self.cc_elar_lambda),
                    prob_channels=int(elar_prob_channels),
                    detach_prob=bool(elar_detach_prob),
                    num_iters=int(elar_num_iters),
                    residual=bool(elar_residual),
                    residual_weight=float(elar_residual_weight),
                )

    @staticmethod
    def _validate_and_normalize_source_indices(
        source_indices: Sequence[Sequence[int]],
        num_stages: int,
    ) -> List[List[int]]:
        """Validate user-specified 2D `source_indices`.

        - Ensures a list of length `num_stages`.
        - Each inner list must be non-empty.
        - Each index must be in [-1, num_stages-1].
        - -1 is not allowed for the coarsest stage (num_stages-1).
        - Duplicates are removed (order preserved).
        """
        if len(source_indices) != num_stages:
            raise ValueError(f'source_indices must have length {num_stages}, got {len(source_indices)}')

        out: List[List[int]] = []
        for t, idxs in enumerate(source_indices):
            if not isinstance(idxs, (list, tuple)):
                raise TypeError(f'source_indices[{t}] must be a list/tuple of ints.')
            if len(idxs) == 0:
                raise ValueError(f'source_indices[{t}] must be non-empty.')

            idxs_u = _unique_preserve_order([int(i) for i in idxs])

            for i in idxs_u:
                if i < -1 or i >= num_stages:
                    raise ValueError(
                        f'Invalid source index {i} in source_indices[{t}]. '
                        f'Valid range is [-1, {num_stages - 1}].',
                    )

            if t == (num_stages - 1) and (-1 in idxs_u):
                raise ValueError(
                    f'source_indices[{t}] (coarsest stage) cannot include -1 because there is no prev-dec.',
                )

            out.append(idxs_u)

        return out

    def _build_fusion_module(
        self,
        in_channels: int,
        out_channels: int,
        fuse_type: str,
        num_convs: int,
        kernel_sizes: Optional[Sequence[int]],
        mid_channels: Optional[int],
        act_last: bool,
    ) -> nn.Module:
        """Build the fusion module used by the multi-stage head.

        Supported `fuse_type`:
          - 'conv1x1': concat -> 1x1 ConvModule
          - 'multi_conv': concat -> conv stack (bottleneck-friendly)
        """
        fuse_type = str(fuse_type).lower()

        if fuse_type in {'1x1', 'conv1x1', 'concat+1x1', 'concat_1x1'}:
            return ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=self.norm_cfg,
            )

        if fuse_type in {'multi_conv', 'conv_stack', 'concat+multi_conv', 'concat_multi_conv'}:
            num_convs = int(num_convs)
            if num_convs < 2:
                raise ValueError('multi_conv fusion requires num_convs >= 2')

            if kernel_sizes is None:
                kernel_sizes = [1, 1] if num_convs == 2 else ([1] + [3] * (num_convs - 2) + [1])
            kernel_sizes = _to_list(kernel_sizes, num_convs)

            if mid_channels is None:
                mid_channels = out_channels

            layers: List[nn.Module] = []
            c_in = in_channels
            for i in range(num_convs):
                k = int(kernel_sizes[i])
                c_out = out_channels if i == (num_convs - 1) else int(mid_channels)
                padding = 0 if k == 1 else (k // 2)

                # Keep activation on intermediate layers; last layer activation is optional
                act_cfg = self.act_cfg if (i < num_convs - 1 or act_last) else None

                layers.append(
                    ConvModule(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=k,
                        stride=1,
                        padding=padding,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg,
                    ),
                )
                c_in = c_out

            return nn.Sequential(*layers)

        raise ValueError(f'Unsupported fuse_type="{fuse_type}". Supported: conv1x1 | multi_conv')

    def _align(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Align `x` to `size`.

        - Downsample: adaptive avg pool / area / bilinear
        - Upsample: `resize` with `interpolate_mode`
        """
        if x.shape[2:] == size:
            return x

        H, W = x.shape[2:]
        th, tw = size

        # Downsample
        if th <= H and tw <= W:
            if self.downsample_mode == 'avg':
                return F.adaptive_avg_pool2d(x, output_size=size)
            if self.downsample_mode == 'area':
                return F.interpolate(x, size=size, mode='area')
            # bilinear
            return resize(x, size=size, mode='bilinear', align_corners=self.align_corners)

        # Upsample
        return resize(x, size=size, mode=self.interpolate_mode, align_corners=self.align_corners)

    def _get_prev_up(self, stage: int, prev: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Resize previous decoded feature to current stage size and match channels."""
        prev_up = resize(prev, size=size, mode=self.interpolate_mode, align_corners=self.align_corners)
        if self.prev_projs is not None:
            prev_up = self.prev_projs[stage](prev_up)
        return prev_up

    def _collect_sources(
        self,
        stage: int,
        proj_feats: List[torch.Tensor],
        prev_up: Optional[torch.Tensor],
        decoded_feats: Sequence[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Collect sources for SSMA at a given stage.

        Sources are collected in two parts:

        1) User-specified `source_indices[stage]`:
             - encoder stages: 0..N-1 (after Align+Project + optional per-source proj)
             - -1: Up(D_{stage+1}) (previous decoded feature)

        2) (A) Optional cross-layer decoder memory:
             - always adds deeper decoded features D_{stage+k} (k>=1) when enabled,
               upsampled to the current stage resolution and projected to stage channels.
             - `cross_layer_kv_depth` controls the maximum hop distance:
                 -1: use all deeper stages
                  0: disable (no-op)
                  1: only include D_{stage+1}
                  2: include D_{stage+1} and D_{stage+2}, etc.

        Note: Because SSMA mixes sources **per pixel**, all sources are aligned to the
        current stage spatial size.
        """
        assert self.source_indices is not None
        assert self.src_projs is not None

        size = proj_feats[stage].shape[2:]
        sources: List[torch.Tensor] = []

        # ---- (1) encoder sources + optional -1 ----
        for idx in self.source_indices[stage]:
            if idx == -1:
                if prev_up is None:
                    raise RuntimeError(
                        f'source_indices[{stage}] includes -1 but prev_up is None. '
                        '(-1 is only valid for non-coarsest stages.)',
                    )
                sources.append(prev_up)
                continue

            # encoder source
            z = self._align(proj_feats[idx], size)
            z = self.src_projs[stage][str(idx)](z)
            sources.append(z)

        # ---- (2) cross-layer decoder memory (A) ----
        if self.use_ssma and self.use_cross_layer_kv and (self.cross_layer_kv_depth != 0):
            # Add D_{stage+1} if not already added via -1.
            if (
                    prev_up is not None
                    and (-1 not in self.source_indices[stage])
                    and (self.cross_layer_kv_depth == -1 or self.cross_layer_kv_depth >= 1)
            ):
                sources.append(prev_up)

            # Add D_{stage+k}, k>=2 (if projections are built)
            if self.dec_projs is not None and (self.cross_layer_kv_depth == -1 or self.cross_layer_kv_depth >= 2):
                num_stages = len(decoded_feats)
                for s in range(stage + 2, num_stages):
                    hop = s - stage
                    if self.cross_layer_kv_depth != -1 and hop > self.cross_layer_kv_depth:
                        break
                    d = decoded_feats[s]
                    if d is None:
                        continue
                    d_up = resize(d, size=size, mode=self.interpolate_mode, align_corners=self.align_corners)
                    d_up = self.dec_projs[stage][str(s)](d_up)
                    sources.append(d_up)

        return sources

    def _decode(self, proj_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """SSMA + PRP decode forward."""
        num_stages = len(proj_feats)
        decoded: List[Optional[torch.Tensor]] = [None for _ in range(num_stages)]

        # Reset per-forward regularization term (for HSSMA + L_div)
        div_loss_total: Optional[torch.Tensor] = None

        stage_aux_logits: Optional[List[Optional[torch.Tensor]]] = None
        if self.use_stage_elar and self.stage_elar_aux_loss:
            stage_aux_logits = [None for _ in range(num_stages)]
        prev: Optional[torch.Tensor] = None
        for stage in reversed(range(num_stages)):
            cur = proj_feats[stage]
            size = cur.shape[2:]

            # We only compute Up(D_{t+1}) when it is needed at this stage:
            #   - PRP query (use_prp=True)
            #   - SSMA sources include -1
            need_prev_here = (prev is not None) and (
                    self.use_prp
                    or (
                            self.use_ssma
                            and self.source_indices is not None
                            and (-1 in self.source_indices[stage])
                    )
                    or (
                            self.use_ssma
                            and self.use_cross_layer_kv
                            and (self.cross_layer_kv_depth != 0)
                            and (self.cross_layer_kv_depth == -1 or self.cross_layer_kv_depth >= 1)
                    )
            )

            prev_up: Optional[torch.Tensor] = None
            if need_prev_here:
                prev_up = self._get_prev_up(stage, prev, size)

            # PRP query: Q_t = LN(Z_tt + Up(D_{t+1}))
            q = cur
            if self.use_prp and (prev_up is not None):
                q = q + prev_up
            q = self.q_norms[stage](q)

            # SSMA decode block
            if self.use_ssma:
                assert self.ssma_blocks is not None
                sources = self._collect_sources(stage, proj_feats, prev_up, decoded)
                block = self.ssma_blocks[stage]
                d = block(q, sources)

                # Accumulate L_div for HSSMA (training only)
                if (
                        self.training
                        and self.ssma_type == 'hssma'
                        and self.hssma_use_div_loss
                        and isinstance(block, HSSMABlock)
                        and (block.last_div_loss is not None)
                ):
                    div_loss_total = (
                        block.last_div_loss if div_loss_total is None else (div_loss_total + block.last_div_loss)
                    )
            else:
                d = q

            # ---- Stage-ELAR (feature refinement) ----
            if self.use_stage_elar and (self.stage_elar_blocks is not None):
                low = self._align(proj_feats[self.stage_elar_low_index], size)
                d = self.stage_elar_blocks[stage](d, low)

                # store aux logits for deep supervision (optional)
                if self.stage_elar_aux_loss and (stage_aux_logits is not None) and (
                        stage in self.stage_elar_aux_stages):
                    blk = self.stage_elar_blocks[stage]
                    if isinstance(blk, StageELARRefine):
                        stage_aux_logits[stage] = blk.last_aux_logits
                else:
                    # drop python ref to save memory
                    blk = self.stage_elar_blocks[stage]
                    if isinstance(blk, StageELARRefine):
                        blk.last_aux_logits = None

            decoded[stage] = d
            prev = d

        decoded_feats: List[torch.Tensor] = [x for x in decoded if x is not None]

        # Store the last div loss for `loss_by_feat` (do NOT detach)
        self._last_div_loss = div_loss_total
        self._last_stage_aux_logits = stage_aux_logits

        # Multi-stage head
        if self.use_multi_stage_fuse:
            assert self.fusion_conv is not None
            outs = []
            tgt_size = decoded_feats[0].shape[2:]
            for x in decoded_feats:
                outs.append(
                    resize(
                        input=x,
                        size=tgt_size,
                        mode=self.interpolate_mode,
                        align_corners=self.align_corners,
                    ),
                )
            fused = self.fusion_conv(torch.cat(outs, dim=1))
        else:
            fused = decoded_feats[0]
            if self.out_proj is not None:
                fused = self.out_proj(fused)

        seg_logits = self.cls_seg(fused)
        return fused, seg_logits, decoded_feats

    def forward(self, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        # Receive multi-stage backbone features (e.g., 1/4, 1/8, 1/16, 1/32)
        inputs = self._transform_inputs(inputs)

        proj_feats = [self.proj_convs[i](inputs[i]) for i in range(len(inputs))]

        fused, seg_logits, decoded_feats = self._decode(proj_feats)

        # ELAR refinement at highest resolution stage
        if self.use_elar and self.elar is not None:
            seg_logits = self.elar(seg_logits, feat_high=decoded_feats[0], feat_low=proj_feats[0])

        return seg_logits

    def loss_by_feat(self, seg_logits: torch.Tensor, batch_data_samples: Any, **kwargs) -> dict:
        """Compute losses from segmentation logits.

        We delegate the standard seg loss/accuracy computation to
        :meth:`BaseDecodeHead.loss_by_feat`, then (optionally) add the HSSMA
        diversity regularization term.
        """
        losses = super().loss_by_feat(seg_logits, batch_data_samples, **kwargs)

        if (self.training
                and self.use_ssma
                and self.ssma_type == 'hssma'
                and self.hssma_use_div_loss
                and (self._last_div_loss is not None)
                and (self.hssma_div_loss_weight != 0.0)):
            losses['loss_div'] = self.hssma_div_loss_weight * self._last_div_loss

        # ---- Stage-ELAR auxiliary losses (optional deep supervision) ----
        if (self.training
                and self.use_stage_elar
                and self.stage_elar_aux_loss
                and (self._last_stage_aux_logits is not None)):
            for t in self.stage_elar_aux_stages:
                aux = self._last_stage_aux_logits[t]
                if aux is None:
                    continue
                w = float(self.stage_elar_aux_weight_list[t])
                if w == 0.0:
                    continue

                aux_losses = super().loss_by_feat(aux, batch_data_samples, **kwargs)
                # keep only loss_* terms and rename to avoid key collision
                for k, v in aux_losses.items():
                    if not k.startswith('loss'):
                        continue
                    losses[f'{k}_stage{t}'] = w * v

        return losses
