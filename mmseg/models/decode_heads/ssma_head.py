# SSMA-ELAR decoder head for MMSegmentation (SegFormer-style backbones).
#
# This file implements the decoder proposed in the accompanying design document:
#   - Align + Project: unify multi-stage encoder features to stage-wise channels
#   - PRP: Progressive Residual Propagation to build per-stage queries
#   - SSMA: Scale-Selective Mix-Attention (softmax / avg scale mixing)
#   - Multi-stage head: upsample decoded features and fuse for segmentation logits
#   - ELAR: Edge-aware Local Affinity Refinement on logits (optional)
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


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x


class MixFFN2D(nn.Module):
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


class WindowCrossAttention(nn.Module):
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


class GlobalCrossAttention(nn.Module):
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

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = int(sr_ratio)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=True)
            self.sr_norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.sr_norm = None

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)

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

        out, _ = self.attn(q_seq, k_seq, v_seq)
        out = self.proj_drop(out)
        out = out.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return out


class SSMABlock(nn.Module):
    """Scale-Selective Mix-Attention block (cross-attn + FFN, no self-attn).

    - Input query `q` is assumed already normalized (PRP does LN outside).
    - Candidate sources are aligned to the same spatial size as `q`, and have
      the same channel dimension `dim`.

    Supported scale-mix modes:
      - softmax: per-pixel softmax over sources (default)
      - avg: uniform average over sources
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

        if use_window_attn:
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

    def _mix_kv(self, q: torch.Tensor, sources: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate K/V from multiple sources using the configured mix mode."""
        if len(sources) == 0:
            raise ValueError('SSMA requires at least one source feature.')

        # Project sources to K/V first
        k_list = [self.k_proj(z) for z in sources]
        v_list = [self.v_proj(z) for z in sources]
        k_stack = torch.stack(k_list, dim=1)  # (B, S, C, H, W)
        v_stack = torch.stack(v_list, dim=1)

        S = k_stack.shape[1]

        if self.mix_mode == 'avg':
            w = torch.full((S,), 1.0 / S, device=q.device, dtype=q.dtype).view(1, S, 1, 1, 1)
            k = (k_stack * w).sum(dim=1)
            v = (v_stack * w).sum(dim=1)
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

        k = (k_stack * alpha).sum(dim=1)
        v = (v_stack * alpha).sum(dim=1)
        return k, v

    def forward(self, q: torch.Tensor, sources: List[torch.Tensor]) -> torch.Tensor:
        # q: (B,C,H,W), sources: list[(B,C,H,W)] aligned
        q_attn = self.q_proj(q)
        k, v = self._mix_kv(q, sources)

        o = self.attn(q_attn, k, v)  # (B,C,H,W)
        x = self.post_attn_norm(q + self.drop_path(o))
        x = x + self.drop_path(self.ffn(x))
        return x


class ELARRefine(nn.Module):
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


@MODELS.register_module()
class SSMAHead(BaseDecodeHead):
    """SSMA-ELAR decoder head.

    This head supports the proposed SSMA + PRP decoding pipeline with optional
    multi-stage fusion and optional ELAR refinement.

    Key requirements for this refactored version:
      - No dict-wrapped configs in __init__ (explicit args)
      - No decode_mode switch (only SSMA-ELAR forward)
      - SSMA mix_mode supports only: softmax | avg
      - Sources are specified by `source_indices` (2D list), no source_mode

    Args:
        interpolate_mode: upsample mode used by `resize`.
        use_ssma: enable SSMA blocks.
        use_prp: enable PRP query construction (residual Up).
        use_multi_stage_fuse: fuse decoded features from all stages.
        use_elar: enable ELAR logit refinement.

        fuse_type: fusion module type for multi-stage fuse. 'conv1x1' or 'multi_conv'.
        fuse_num_convs: number of conv layers when fuse_type='multi_conv'.
        fuse_kernel_sizes: per-layer kernel sizes for 'multi_conv'.
        fuse_mid_channels: intermediate channels for 'multi_conv'.
        fuse_act_last: whether to apply activation on the last fusion layer.

        source_indices: 2D list specifying sources per stage. Use -1 to include prev-dec.

        downsample_mode: how to align encoder features when downsampling.
            Options: 'avg' | 'bilinear' | 'area'.

        ssma_*: SSMA block hyper-parameters.
        elar_*: ELAR hyper-parameters.

        **kwargs: BaseDecodeHead args (in_channels, channels, num_classes, etc.).
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        # Ablation switches
        use_ssma: bool = True,
        use_prp: bool = True,
        use_multi_stage_fuse: bool = True,
        use_elar: bool = True,
        # Multi-stage fusion
        fuse_type: str = 'conv1x1',
        fuse_num_convs: int = 3,
        fuse_kernel_sizes: Optional[Sequence[int]] = None,
        fuse_mid_channels: Optional[int] = None,
        fuse_act_last: bool = False,
        # SSMA: source selection & alignment
        source_indices: Optional[Sequence[Sequence[int]]] = None,
        downsample_mode: str = 'bilinear',
        # SSMA: attention + FFN
        ssma_num_heads: Union[int, Sequence[int]] = 8,
        ssma_window_size: Union[int, Sequence[int]] = 7,
        ssma_use_window_attn: Union[bool, Sequence[bool]] = False,
        ssma_use_rel_pos_bias: bool = True,
        ssma_sr_ratio: Union[int, Sequence[int]] = 1,
        ssma_mix_mode: str = 'softmax',
        ssma_gate_channels: int = 64,
        ssma_mlp_ratio: Union[int, Sequence[int]] = 4,
        ssma_attn_drop: float = 0.0,
        ssma_proj_drop: float = 0.0,
        ssma_ffn_drop: float = 0.0,
        ssma_drop_path: Union[float, Sequence[float]] = 0.0,
        # ELAR
        elar_kernel_size: int = 5,
        elar_guidance_channels: int = 64,
        elar_theta_channels: int = 16,
        elar_use_prob: bool = True,
        elar_prob_channels: int = 16,
        elar_detach_prob: bool = True,
        elar_num_iters: int = 1,
        elar_residual: bool = True,
        elar_residual_weight: float = 1.0,
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

        # BaseDecodeHead expects an int `channels` to build cls_seg.
        kwargs['channels'] = int(self.stage_channels[0])

        super().__init__(input_transform='multiple_select', **kwargs)

        # ---- Store switches / common settings ----
        self.interpolate_mode = str(interpolate_mode)
        self.use_ssma = bool(use_ssma)
        self.use_prp = bool(use_prp)
        self.use_multi_stage_fuse = bool(use_multi_stage_fuse)
        self.use_elar = bool(use_elar)

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
                ssma_blocks.append(
                    SSMABlock(
                        dim=c,
                        num_heads=h,
                        window_size=int(window_size[t]),
                        gate_channels=int(ssma_gate_channels),
                        mlp_ratio=int(mlp_ratio[t]),
                        mix_mode=str(ssma_mix_mode),
                        use_window_attn=bool(use_window_attn[t]),
                        use_rel_pos_bias=bool(ssma_use_rel_pos_bias),
                        sr_ratio=int(sr_ratio[t]),
                        attn_drop=float(ssma_attn_drop),
                        proj_drop=float(ssma_proj_drop),
                        ffn_drop=float(ssma_ffn_drop),
                        drop_path=float(drop_path[t]),
                    ),
                )
            self.ssma_blocks = ssma_blocks

        # ---- Multi-stage fusion conv (only if enabled) ----
        self.fusion_conv: Optional[nn.Module] = None
        if self.use_multi_stage_fuse:
            fuse_in_channels = int(sum(self.stage_channels))
            fuse_out_channels = int(self.stage_channels[0])
            self.fusion_conv = self._build_fusion_module(
                in_channels=fuse_in_channels,
                out_channels=fuse_out_channels,
                fuse_type=fuse_type,
                num_convs=fuse_num_convs,
                kernel_sizes=fuse_kernel_sizes,
                mid_channels=fuse_mid_channels,
                act_last=fuse_act_last,
            )

        # ---- ELAR refinement (only if enabled) ----
        self.elar: Optional[ELARRefine] = None
        if self.use_elar:
            self.elar = ELARRefine(
                num_logits_channels=self.out_channels,
                feat_channels=int(self.channels),
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
    ) -> List[torch.Tensor]:
        """Collect and align sources for SSMA at a given stage."""
        assert self.source_indices is not None
        assert self.src_projs is not None

        size = proj_feats[stage].shape[2:]
        sources: List[torch.Tensor] = []

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

        return sources

    def _decode(self, proj_feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """SSMA + PRP decode forward."""
        num_stages = len(proj_feats)
        decoded: List[Optional[torch.Tensor]] = [None for _ in range(num_stages)]

        prev: Optional[torch.Tensor] = None
        for stage in reversed(range(num_stages)):
            cur = proj_feats[stage]
            size = cur.shape[2:]

            # We only compute Up(D_{t+1}) when it is needed at this stage:
            #   - PRP query (use_prp=True)
            #   - SSMA sources include -1
            need_prev_here = (prev is not None) and (
                    self.use_prp or (
                    self.use_ssma and self.source_indices is not None and (-1 in self.source_indices[stage]))
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
                sources = self._collect_sources(stage, proj_feats, prev_up)
                d = self.ssma_blocks[stage](q, sources)
            else:
                d = q

            decoded[stage] = d
            prev = d

        decoded_feats: List[torch.Tensor] = [x for x in decoded if x is not None]

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
