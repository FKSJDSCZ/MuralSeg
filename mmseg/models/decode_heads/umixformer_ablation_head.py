# ---------------------------------------------------------------
# Refactored U-MixFormer decoder head with optional HSSMA + ELAR.
#
# This file is designed for MMSegmentation (MMEngine-based) and keeps the
# original APFormerHead2 (U-MixFormer) decoding logic as the **baseline**.
#
# Ablations supported via init args (no nested dict):
#   1) baseline U-MixFormer:               use_hssma=False, use_elar=False
#   2) replace attn+ffn with HSSMA system: use_hssma=True,  use_elar=False
#   3) U-MixFormer + ELAR:                 use_hssma=False, use_elar=True
#   4) U-MixFormer + ELAR + HSSMA:         use_hssma=True,  use_elar=True
#
# Stage-wise list args MUST be in order: s4 -> s1 (deep -> shallow).
#   - s4 corresponds to c4 (lowest resolution)
#   - s1 corresponds to c1 (highest resolution)
# ---------------------------------------------------------------

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


def _to_stage_list(x: Union[Any, Sequence[Any]], length: int = 4, name: str = '') -> List[Any]:
    """Broadcast scalar to a fixed-length stage list.

    Args:
        x: scalar or list/tuple.
        length: number of stages (fixed to 4).
        name: optional name for error messages.
    """
    if isinstance(x, (list, tuple)):
        if len(x) != length:
            raise ValueError(f'{name} must have length {length} (s4->s1), but got {len(x)}')
        return list(x)
    return [x for _ in range(length)]


# -----------------------------------------------------------------------------
# U-MixFormer baseline blocks (token/NLC style; refactored for readability)
# -----------------------------------------------------------------------------


class DWConv(BaseModule):
    """Depth-wise 3x3 conv used inside the MLP (token -> feature -> token)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixMLP(BaseModule):
    """U-MixFormer MLP (Linear -> DWConv -> GELU -> Linear)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = int(out_features or in_features)
        hidden_features = int(hidden_features or in_features)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModule):
    """Pool + 1x1 conv each stage feature then concatenate along channels.

    Notes:
        - This matches the official APFormerHead2 logic.
        - `pool_ratio` is specified in order s4->s1 and applied to [c4,c3,c2,c1].
    """

    def __init__(self, pool_ratio: Sequence[int], dims_s4s1: Sequence[int]):
        super().__init__()
        if len(pool_ratio) != 4 or len(dims_s4s1) != 4:
            raise ValueError('CatKey expects 4-stage pool_ratio and dims in order [c4,c3,c2,c1].')

        self.pool_ratio = [int(x) for x in pool_ratio]
        self._sr_convs = ModuleList()
        self._pools = ModuleList()
        for r, c in zip(self.pool_ratio, dims_s4s1):
            if r > 1:
                self._sr_convs.append(nn.Conv2d(c, c, kernel_size=1, stride=1, bias=True))
                self._pools.append(nn.AvgPool2d(kernel_size=r, stride=r, ceil_mode=True))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, feats_s4s1: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats_s4s1) != 4:
            raise ValueError('CatKey forward expects 4 tensors: [c4,c3,c2,c1].')

        out: List[torch.Tensor] = []
        cnt = 0
        for i, r in enumerate(self.pool_ratio):
            x = feats_s4s1[i]
            if r > 1:
                x = self._sr_convs[cnt](self._pools[cnt](x))
                cnt += 1
            out.append(x)
        return torch.cat(out, dim=1)


class CrossAttention(BaseModule):
    """Cross-attention (query=x, key/value=y) used in official U-MixFormer head.

    Important:
        The official APFormerHead2 code constructs `pool`/`sr`/`norm` but
        effectively only applies GELU to `y` in forward (pooling is commented out,
        and norm output is overwritten). We keep the same behavior for baseline
        compatibility.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pool_ratio: int = 16,
    ):
        super().__init__()
        if dim_q % num_heads != 0:
            raise ValueError(f'dim_q={dim_q} must be divisible by num_heads={num_heads}')

        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        self.pool_ratio = int(pool_ratio)

        head_dim = self.dim_q // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(self.dim_q, self.dim_q, bias=qkv_bias)
        self.kv = nn.Linear(self.dim_kv, self.dim_q * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_q, self.dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        # Unused (kept for checkpoint compatibility with the official impl)
        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            # self.sr = nn.Conv2d(self.dim_kv, self.dim_kv, kernel_size=1, stride=1)
        # self.norm = nn.LayerNorm(self.dim_kv)
        self.act = nn.GELU()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor, H2: int, W2: int) -> torch.Tensor:
        # x: (B, Nq, Cq), y: (B, Nk, Ckv)
        B, Nq, Cq = x.shape
        _, Nk, Ckv = y.shape

        q = self.q(x).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)

        # Official behavior: effectively `x_ = GELU(y)`.
        # (Pooling and norm exist but have no effect on the output.)
        x_ = self.act(y) if self.pool_ratio >= 0 else y

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, Cq // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class UMixFormerBlock(BaseModule):
    """(Norm -> CrossAttn -> Norm -> MLP) block used in APFormerHead2."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        pool_ratio: int = 16,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_kv)
        self.norm3 = nn.LayerNorm(dim_q)

        self.attn = CrossAttention(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            pool_ratio=pool_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = MixMLP(in_features=dim_q, hidden_features=mlp_hidden_dim, drop=drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor, H2: int, W2: int, H1: int, W1: int) -> torch.Tensor:
        # Keep the exact residual / pre-norm structure of the official impl:
        #   x = x + Attn(LN(x), LN(y))
        #   x = x + MLP(LN(x))
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2))
        x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))
        return x


# -----------------------------------------------------------------------------
# HSSMA (+L_div) and ELAR blocks extracted from ssma_head_v2.py
#   - Strip attention code is removed.
# -----------------------------------------------------------------------------


class ChannelLayerNorm(BaseModule):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class MixFFN2D(BaseModule):
    """SegFormer-style MixFFN for 2D features: 1x1 -> DWConv3x3 -> GELU -> 1x1."""

    def __init__(self, embed_dims: int, feedforward_channels: int, ffn_drop: float = 0.0):
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
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalCrossAttention(BaseModule):
    """Global cross-attention with optional spatial reduction (sr_ratio) on K/V.

    Q/K/V are assumed to be already projected by the caller.
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
            self.sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=True)
            self.sr_norm = nn.LayerNorm(self.dim)
        else:
            self.sr = None
            self.sr_norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

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


class HSSMABlock(BaseModule):
    """Head-wise Scale-Selective Mix-Attention (HSSMA) block (strip-attn removed).

    This block:
      1) mixes multi-source K/V **per head per pixel** via a softmax gate
      2) applies global cross-attention Q -> mixed(K,V)
      3) applies a MixFFN

    Diversity loss (L_div): optional regularizer on head-wise mixing weights.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        mix_mode: str = 'softmax',
        gate_channels: int = 64,
        mlp_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        drop_path: float = 0.0,
        compute_div_loss: bool = False,
        div_loss_max_samples: int = 65536,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim {dim} must be divisible by num_heads {num_heads}')
        mix_mode = str(mix_mode).lower()
        if mix_mode not in {'softmax', 'avg'}:
            raise ValueError("mix_mode must be 'softmax' or 'avg'")
        if gate_channels % num_heads != 0:
            raise ValueError('gate_channels must be divisible by num_heads')

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads

        self.mix_mode = mix_mode
        self.gate_channels = int(gate_channels)
        self.gate_channels_per_head = self.gate_channels // self.num_heads

        self.q_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=True)
        self.gate_proj = nn.Conv2d(self.dim, self.gate_channels, kernel_size=1, bias=True)

        self.attn = GlobalCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            sr_ratio=int(sr_ratio),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.post_attn_norm = ChannelLayerNorm(self.dim)
        self.ffn = MixFFN2D(self.dim, feedforward_channels=int(mlp_ratio) * self.dim, ffn_drop=ffn_drop)

        # L_div
        self.compute_div_loss = bool(compute_div_loss)
        self.div_loss_max_samples = int(div_loss_max_samples)
        self.last_div_loss: Optional[torch.Tensor] = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

    def _sample_tokens(self, a: torch.Tensor) -> torch.Tensor:
        # a: (N, heads, S)
        if self.div_loss_max_samples <= 0:
            return a
        N = a.shape[0]
        if N <= self.div_loss_max_samples:
            return a
        idx = torch.randperm(N, device=a.device)[: self.div_loss_max_samples]
        return a.index_select(0, idx)

    def _compute_div_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        # alpha: (B, heads, S, H, W)
        B, Hh, S, H, W = alpha.shape
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

        k_list = [self.k_proj(z) for z in sources]
        v_list = [self.v_proj(z) for z in sources]
        k_stack = torch.stack(k_list, dim=1).view(B, S, self.num_heads, self.head_dim, H, W)
        v_stack = torch.stack(v_list, dim=1).view(B, S, self.num_heads, self.head_dim, H, W)

        if self.mix_mode == 'avg':
            alpha = torch.full(
                (B, self.num_heads, S, H, W),
                1.0 / float(S),
                device=q.device,
                dtype=q.dtype,
            )
        else:
            q_gate = self.gate_proj(q).view(B, self.num_heads, self.gate_channels_per_head, H, W)
            scores = []
            for z in sources:
                z_gate = self.gate_proj(z).view(B, self.num_heads, self.gate_channels_per_head, H, W)
                s = (q_gate * z_gate).sum(dim=2) / math.sqrt(self.gate_channels_per_head)
                scores.append(s.unsqueeze(2))
            scores = torch.cat(scores, dim=2)  # (B, heads, S, H, W)
            alpha = scores.softmax(dim=2)

        a = alpha.permute(0, 2, 1, 3, 4).unsqueeze(3)  # (B,S,heads,1,H,W)
        k_mix = (k_stack * a).sum(dim=1).reshape(B, C, H, W)
        v_mix = (v_stack * a).sum(dim=1).reshape(B, C, H, W)

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
    """Edge-aware Local Affinity Refinement (ELAR) on logits."""

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

        in_g = int(feat_channels) * 2
        if self.use_prob:
            self.prob_proj = nn.Conv2d(num_logits_channels, prob_channels, kernel_size=1, bias=True)
            in_g = in_g + int(prob_channels)
        else:
            self.prob_proj = None

        self.guidance_fuse = nn.Conv2d(in_g, int(guidance_channels), kernel_size=1, bias=True)
        self.guidance_norm = ChannelLayerNorm(int(guidance_channels))
        self.theta = nn.Conv2d(int(guidance_channels), self.theta_channels, kernel_size=1, bias=True)

        self.rel_bias = nn.Parameter(torch.zeros(self.kernel_size * self.kernel_size))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0.0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)

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
            g_list = [feat_high, feat_low]
            if self.use_prob:
                p = self._prob(x)
                if self.detach_prob:
                    p = p.detach()
                p = self.prob_proj(p)
                g_list.append(p)

            g = torch.cat(g_list, dim=1)
            g = self.guidance_norm(self.guidance_fuse(g))
            t = self.theta(g)

            t_unf = F.unfold(t, kernel_size=k, padding=pad)
            t_unf = t_unf.view(B, self.theta_channels, k * k, H * W).permute(0, 3, 2, 1).contiguous()
            t_center = t.flatten(2).transpose(1, 2).contiguous()

            scores = (t_unf * t_center.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.theta_channels)
            scores = scores + self.rel_bias.view(1, 1, k * k).to(dtype=scores.dtype)
            weights = scores.softmax(dim=-1)

            l_unf = F.unfold(x, kernel_size=k, padding=pad)
            l_unf = l_unf.view(B, C_out, k * k, H * W).permute(0, 3, 2, 1).contiguous()
            x_ref = (weights.unsqueeze(-1) * l_unf).sum(dim=2)
            x_ref = x_ref.transpose(1, 2).reshape(B, C_out, H, W).contiguous()

            if self.residual:
                x = x + self.residual_weight * (x_ref - x)
            else:
                x = x_ref

        return x


# -----------------------------------------------------------------------------
# Refactored 4-stage U-MixFormer decoder head with ablations
# -----------------------------------------------------------------------------


@MODELS.register_module()
class UMixFormerAblationHead(BaseDecodeHead):
    """U-MixFormer (APFormerHead2-style) decode head with optional HSSMA + ELAR.

    Notes:
        - Decoder is fixed to 4 stages.
        - Stage-wise lists must be in order s4->s1 (deep->shallow).
        - `in_channels` order in MMSeg is typically [c1,c2,c3,c4] (shallow->deep).
    """

    def __init__(
        self,
        feature_strides: Sequence[int],
        embed_dim: int,
        # Ablation switches
        use_hssma: bool = False,
        use_elar: bool = False,
        # ---- Baseline U-MixFormer params ----
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        attn_pool_ratio: Sequence[int] = (8, 4, 2, 1),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.1,
        # ---- HSSMA params ----
        hssma_num_heads: Optional[Sequence[int]] = None,
        hssma_sr_ratio: Sequence[int] = (1, 2, 4, 8),
        hssma_mix_mode: str = 'softmax',
        hssma_gate_channels: int = 64,
        hssma_mlp_ratio: Union[int, Sequence[int]] = 4,
        hssma_attn_drop: float = 0.0,
        hssma_proj_drop: float = 0.0,
        hssma_ffn_drop: float = 0.0,
        hssma_drop_path_rate: Union[float, Sequence[float]] = 0.1,
        # HSSMA + L_div
        hssma_use_div_loss: bool = False,
        hssma_div_loss_weight: float = 0.0,
        hssma_div_loss_max_samples: int = 65536,
        # ---- Feature alignment for HSSMA sources ----
        downsample_mode: str = 'avg',
        interpolate_mode: str = 'bilinear',
        # ---- ELAR ----
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
        # ---- validate fixed-4-stage assumptions ----
        raw_in_channels = kwargs.get('in_channels', None)
        if not isinstance(raw_in_channels, (list, tuple)) or len(raw_in_channels) != 4:
            raise ValueError('UMixFormerAblationHead requires in_channels as a list/tuple of 4 ints: [c1,c2,c3,c4].')

        if len(feature_strides) != 4:
            raise ValueError('feature_strides must have length 4 (for c1..c4).')

        # BaseDecodeHead builds classifier using `channels`.
        if 'channels' in kwargs and kwargs['channels'] != embed_dim:
            raise ValueError(
                f'Got channels={kwargs["channels"]} but embed_dim={embed_dim}. '
                'Please set channels==embed_dim in the decode_head config.',
            )
        kwargs['channels'] = int(embed_dim)

        super().__init__(input_transform='multiple_select', **kwargs)

        self.feature_strides = tuple(int(s) for s in feature_strides)
        self.use_hssma = bool(use_hssma)
        self.use_elar = bool(use_elar)

        self.interpolate_mode = str(interpolate_mode)
        self.downsample_mode = str(downsample_mode)
        if self.downsample_mode not in {'avg', 'area', 'bilinear'}:
            raise ValueError("downsample_mode must be one of {'avg','area','bilinear'}")

        # Stage dims in order s4->s1 (deep->shallow)
        c1, c2, c3, c4 = [int(x) for x in self.in_channels]
        self.dims_s4s1 = [c4, c3, c2, c1]
        self.tot_channels = int(sum(self.in_channels))
        self.embed_dim = int(embed_dim)

        # ---- fuse (always used) ----
        self.fusion_conv = ConvModule(
            in_channels=self.tot_channels,
            out_channels=self.embed_dim,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
        )

        # ---- ELAR (optional) ----
        self.elar: Optional[ELARRefine]
        if self.use_elar:
            # Use highest-resolution stage (c1) features for guidance.
            self.elar = ELARRefine(
                num_logits_channels=self.num_classes,
                feat_channels=c1,
                kernel_size=elar_kernel_size,
                guidance_channels=elar_guidance_channels,
                theta_channels=elar_theta_channels,
                use_prob=elar_use_prob,
                prob_channels=elar_prob_channels,
                detach_prob=elar_detach_prob,
                num_iters=elar_num_iters,
                residual=elar_residual,
                residual_weight=elar_residual_weight,
            )
        else:
            self.elar = None

        # ---- Baseline blocks OR HSSMA blocks (mutually exclusive instantiation) ----
        self._last_div_loss: Optional[torch.Tensor] = None
        self.hssma_use_div_loss = bool(hssma_use_div_loss)
        self.hssma_div_loss_weight = float(hssma_div_loss_weight)
        self.hssma_div_loss_max_samples = int(hssma_div_loss_max_samples)

        if not self.use_hssma:
            # ----- U-MixFormer baseline -----
            num_heads_s4s1 = _to_stage_list(num_heads, 4, 'num_heads')
            attn_pool_ratio_s4s1 = _to_stage_list(attn_pool_ratio, 4, 'attn_pool_ratio')
            drop_path_s4s1 = _to_stage_list(drop_path_rate, 4, 'drop_path_rate')

            self.stage_blocks = ModuleList(
                [
                    UMixFormerBlock(
                        dim_q=self.dims_s4s1[i],
                        dim_kv=self.tot_channels,
                        num_heads=int(num_heads_s4s1[i]),
                        mlp_ratio=float(mlp_ratio),
                        drop=float(drop),
                        attn_drop=float(attn_drop),
                        drop_path=float(drop_path_s4s1[i]),
                        pool_ratio=int(attn_pool_ratio_s4s1[i]),
                    )
                    for i in range(4)
                ],
            )

            pool_ratio_s4s1 = _to_stage_list(pool_ratio, 4, 'pool_ratio')
            self.cat_keys = ModuleList(
                [
                    CatKey(pool_ratio=pool_ratio_s4s1, dims_s4s1=self.dims_s4s1) for _ in range(4)
                ],
            )

            # HSSMA-only modules are not instantiated
            self.q_norms = None
            self.hssma_src_projs = None
        else:
            # ----- HSSMA replacement -----
            hssma_num_heads_s4s1 = _to_stage_list(
                hssma_num_heads if hssma_num_heads is not None else num_heads,
                4,
                'hssma_num_heads',
            )
            hssma_sr_ratio_s4s1 = _to_stage_list(hssma_sr_ratio, 4, 'hssma_sr_ratio')
            hssma_mlp_ratio_s4s1 = _to_stage_list(hssma_mlp_ratio, 4, 'hssma_mlp_ratio')
            hssma_drop_path_s4s1 = _to_stage_list(hssma_drop_path_rate, 4, 'hssma_drop_path_rate')

            self.q_norms = ModuleList([ChannelLayerNorm(d) for d in self.dims_s4s1])
            self.stage_blocks = ModuleList(
                [
                    HSSMABlock(
                        dim=self.dims_s4s1[i],
                        num_heads=int(hssma_num_heads_s4s1[i]),
                        sr_ratio=int(hssma_sr_ratio_s4s1[i]),
                        mix_mode=str(hssma_mix_mode),
                        gate_channels=int(hssma_gate_channels),
                        mlp_ratio=int(hssma_mlp_ratio_s4s1[i]),
                        attn_drop=float(hssma_attn_drop),
                        proj_drop=float(hssma_proj_drop),
                        ffn_drop=float(hssma_ffn_drop),
                        drop_path=float(hssma_drop_path_s4s1[i]),
                        compute_div_loss=self.hssma_use_div_loss,
                        div_loss_max_samples=self.hssma_div_loss_max_samples,
                    )
                    for i in range(4)
                ],
            )

            # Per-stage per-source 1x1 projections: [stage][source]
            # - stage order: s4->s1
            # - source order: s4->s1 (i.e. [c4,c3,c2,c1] channels)
            self.hssma_src_projs = ModuleList()
            for t in range(4):
                tgt_c = self.dims_s4s1[t]
                projs_t = ModuleList()
                for src_c in self.dims_s4s1:
                    projs_t.append(
                        ConvModule(
                            in_channels=int(src_c),
                            out_channels=int(tgt_c),
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            norm_cfg=self.norm_cfg,
                            act_cfg=None,
                        ),
                    )
                self.hssma_src_projs.append(projs_t)

            # Baseline-only CatKey modules are not instantiated
            self.cat_keys = None

    def _align(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Align `x` to `size`.

        Downsample uses `downsample_mode`:
          - 'avg': adaptive avg pooling
          - 'area': area interpolation
          - 'bilinear': bilinear interpolation

        Upsample uses `interpolate_mode` (default bilinear).
        """
        if x.shape[2:] == size:
            return x

        H, W = x.shape[2:]
        th, tw = size

        if th <= H and tw <= W:
            # downsample
            if self.downsample_mode == 'avg':
                return F.adaptive_avg_pool2d(x, output_size=size)
            if self.downsample_mode == 'area':
                return F.interpolate(x, size=size, mode='area')
            # bilinear
            return resize(x, size=size, mode='bilinear', align_corners=self.align_corners)

        # upsample
        return resize(x, size=size, mode=self.interpolate_mode, align_corners=self.align_corners)

    def _decode_baseline(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> List[
        torch.Tensor]:
        """Baseline APFormerHead2 decode. Returns decoded features in order [d4,d3,d2,d1]."""
        assert self.cat_keys is not None

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # ---- stage s4 ----
        key = self.cat_keys[0]([c4, c3, c2, c1]).flatten(2).transpose(1, 2)  # (B, Nk, tot)
        q4 = c4.flatten(2).transpose(1, 2)
        d4 = self.stage_blocks[0](q4, key, h4, w4, h4, w4).transpose(1, 2).reshape(n, -1, h4, w4)

        # ---- stage s3 ----
        key = self.cat_keys[1]([d4, c3, c2, c1]).flatten(2).transpose(1, 2)
        q3 = c3.flatten(2).transpose(1, 2)
        d3 = self.stage_blocks[1](q3, key, h4, w4, h3, w3).transpose(1, 2).reshape(n, -1, h3, w3)

        # ---- stage s2 ----
        key = self.cat_keys[2]([d4, d3, c2, c1]).flatten(2).transpose(1, 2)
        q2 = c2.flatten(2).transpose(1, 2)
        d2 = self.stage_blocks[2](q2, key, h4, w4, h2, w2).transpose(1, 2).reshape(n, -1, h2, w2)

        # ---- stage s1 ----
        key = self.cat_keys[3]([d4, d3, d2, c1]).flatten(2).transpose(1, 2)
        q1 = c1.flatten(2).transpose(1, 2)
        d1 = self.stage_blocks[3](q1, key, h4, w4, h1, w1).transpose(1, 2).reshape(n, -1, h1, w1)

        return [d4, d3, d2, d1]

    def _decode_hssma(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> List[
        torch.Tensor]:
        """HSSMA replacement decode. Returns decoded features in order [d4,d3,d2,d1]."""
        assert self.q_norms is not None
        assert self.hssma_src_projs is not None

        # Reset per-forward div loss accumulator (avg over stages with valid div loss)
        div_loss_total: Optional[torch.Tensor] = None
        div_loss_count: int = 0

        # ---- stage s4 (idx=0) ----
        size4 = c4.shape[2:]
        q4 = self.q_norms[0](c4)
        src4 = [
            self.hssma_src_projs[0][0](self._align(c4, size4)),
            self.hssma_src_projs[0][1](self._align(c3, size4)),
            self.hssma_src_projs[0][2](self._align(c2, size4)),
            self.hssma_src_projs[0][3](self._align(c1, size4)),
        ]
        d4 = self.stage_blocks[0](q4, src4)
        if (
                self.training
                and self.hssma_use_div_loss
                and isinstance(self.stage_blocks[0], HSSMABlock)
                and (self.stage_blocks[0].last_div_loss is not None)
        ):
            div_loss_total = self.stage_blocks[0].last_div_loss
            div_loss_count = 1

        # ---- stage s3 (idx=1) ----
        size3 = c3.shape[2:]
        q3 = self.q_norms[1](c3)
        src3 = [
            self.hssma_src_projs[1][0](self._align(d4, size3)),
            self.hssma_src_projs[1][1](self._align(c3, size3)),
            self.hssma_src_projs[1][2](self._align(c2, size3)),
            self.hssma_src_projs[1][3](self._align(c1, size3)),
        ]
        d3 = self.stage_blocks[1](q3, src3)
        if (
                self.training
                and self.hssma_use_div_loss
                and isinstance(self.stage_blocks[1], HSSMABlock)
                and (self.stage_blocks[1].last_div_loss is not None)
        ):
            div_loss_total = (
                self.stage_blocks[1].last_div_loss
                if div_loss_total is None
                else (div_loss_total + self.stage_blocks[1].last_div_loss)
            )
            div_loss_count += 1

        # ---- stage s2 (idx=2) ----
        size2 = c2.shape[2:]
        q2 = self.q_norms[2](c2)
        src2 = [
            self.hssma_src_projs[2][0](self._align(d4, size2)),
            self.hssma_src_projs[2][1](self._align(d3, size2)),
            self.hssma_src_projs[2][2](self._align(c2, size2)),
            self.hssma_src_projs[2][3](self._align(c1, size2)),
        ]
        d2 = self.stage_blocks[2](q2, src2)
        if (
                self.training
                and self.hssma_use_div_loss
                and isinstance(self.stage_blocks[2], HSSMABlock)
                and (self.stage_blocks[2].last_div_loss is not None)
        ):
            div_loss_total = (
                self.stage_blocks[2].last_div_loss
                if div_loss_total is None
                else (div_loss_total + self.stage_blocks[2].last_div_loss)
            )
            div_loss_count += 1

        # ---- stage s1 (idx=3) ----
        size1 = c1.shape[2:]
        q1 = self.q_norms[3](c1)
        src1 = [
            self.hssma_src_projs[3][0](self._align(d4, size1)),
            self.hssma_src_projs[3][1](self._align(d3, size1)),
            self.hssma_src_projs[3][2](self._align(d2, size1)),
            self.hssma_src_projs[3][3](self._align(c1, size1)),
        ]
        d1 = self.stage_blocks[3](q1, src1)
        if (
                self.training
                and self.hssma_use_div_loss
                and isinstance(self.stage_blocks[3], HSSMABlock)
                and (self.stage_blocks[3].last_div_loss is not None)
        ):
            div_loss_total = (
                self.stage_blocks[3].last_div_loss
                if div_loss_total is None
                else (div_loss_total + self.stage_blocks[3].last_div_loss)
            )
            div_loss_count += 1

        # Store for loss_by_feat (average to keep scale stable)
        if div_loss_total is not None and div_loss_count > 0:
            self._last_div_loss = div_loss_total / float(div_loss_count)
        else:
            self._last_div_loss = None
        return [d4, d3, d2, d1]

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        inputs = self._transform_inputs(inputs)
        if len(inputs) != 4:
            raise ValueError(f'Expected 4 input feature maps, got {len(inputs)}')

        c1, c2, c3, c4 = inputs
        if not self.use_hssma:
            decoded = self._decode_baseline(c1, c2, c3, c4)
            # baseline: no div loss
            self._last_div_loss = None
        else:
            decoded = self._decode_hssma(c1, c2, c3, c4)

        d4, d3, d2, d1 = decoded

        # Multi-stage fusion at the highest resolution (c1)
        tgt_size = c1.shape[2:]
        d4u = resize(d4, size=tgt_size, mode=self.interpolate_mode, align_corners=self.align_corners)
        d3u = resize(d3, size=tgt_size, mode=self.interpolate_mode, align_corners=self.align_corners)
        d2u = resize(d2, size=tgt_size, mode=self.interpolate_mode, align_corners=self.align_corners)

        fused = self.fusion_conv(torch.cat([d4u, d3u, d2u, d1], dim=1))
        seg_logits = self.cls_seg(fused)

        # Optional ELAR refinement on logits (guided by stage1 feature + low-level feature)
        if self.use_elar and (self.elar is not None):
            seg_logits = self.elar(seg_logits, feat_high=d1, feat_low=c1)

        return seg_logits

    def loss_by_feat(self, seg_logits: torch.Tensor, batch_data_samples: Any, **kwargs) -> dict:
        losses = super().loss_by_feat(seg_logits, batch_data_samples, **kwargs)

        if (
                self.training
                and self.use_hssma
                and self.hssma_use_div_loss
                and (self._last_div_loss is not None)
                and (self.hssma_div_loss_weight != 0.0)
        ):
            losses['loss_div'] = self.hssma_div_loss_weight * self._last_div_loss

        return losses
