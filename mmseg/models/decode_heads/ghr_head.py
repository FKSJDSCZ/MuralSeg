# ================================================================
# GHR-MixFormer Head
# Key ideas:
#   - Stage-Adaptive Gated KV Fusion (optional)
#   - Hybrid KV tokenization: grid tokens + strip tokens (optional)
#   - Rank-r compressed cross-attention (optional, rank>=1)
#   - Directional Dilated Local Perception Unit (optional)
#   - Gated DWConv-FFN (optional)
# ================================================================

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nchw_to_nlc, nlc_to_nchw


class DWConv(BaseModule):
    """Depthwise 3x3 conv in token space."""

    def __init__(self, dim: int, init_cfg=None):
        super().__init__(init_cfg)
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

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

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = hw_shape
        x = nlc_to_nchw(x, (h, w))
        x = self.dwconv(x)
        x = nchw_to_nlc(x)
        return x


class MlpDWConv(BaseModule):
    """U-MixFormer style FFN: Linear -> DWConv -> GELU -> Linear."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)

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

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, hw_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedDWConvFFN(BaseModule):
    """Gated FFN (GEGLU-like) + DWConv.
    x -> (W_u x) ⊙ GELU(W_v x) -> DWConv -> W_o
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        hidden_dim = int(dim * mlp_ratio)
        self.fc_u = nn.Linear(dim, hidden_dim)
        self.fc_v = nn.Linear(dim, hidden_dim)
        self.dwconv = DWConv(hidden_dim)
        self.drop = nn.Dropout(drop)
        self.fc_out = nn.Linear(hidden_dim, dim)

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

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        u = self.fc_u(x)
        v = self.fc_v(x)
        x = u * F.gelu(v)
        x = self.dwconv(x, hw_shape)
        x = self.drop(x)
        x = self.fc_out(x)
        x = self.drop(x)
        return x


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


class DirectionalDilatedLPU(BaseModule):
    """Directional + dilated local perception, depthwise for efficiency.

    This module is inspired by SCASeg's Local Perception Module (LPM),
    but adds multi-dilation and optional directional kernels.
    """

    def __init__(
        self,
        dim: int,
        dilations: Sequence[int] = (1, 2),
        use_directional: bool = True,
        directional_kernel: int = 7,
        se_reduction: int = 4,
        act_layer: nn.Module = nn.GELU,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.use_directional = use_directional
        self.dilations = tuple(int(d) for d in dilations)

        # Pointwise mixing (kept lightweight; could be groups=dim if you want purely depthwise)
        self.pw1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.act = act_layer()

        # Multi-dilation depthwise conv branches
        self.dw_branches = ModuleList()
        for d in self.dilations:
            self.dw_branches.append(
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=d, dilation=d, groups=dim, bias=True),
            )

        # Optional directional depthwise branches (1 x k, k x 1)
        if self.use_directional:
            k = int(directional_kernel)
            pad = k // 2
            self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, k), stride=1, padding=(0, pad), groups=dim, bias=True)
            self.dw_v = nn.Conv2d(dim, dim, kernel_size=(k, 1), stride=1, padding=(pad, 0), groups=dim, bias=True)

        # Channel gate (SE-style, very cheap)
        mid = max(dim // int(se_reduction), 4)
        self.se = Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=True),
            act_layer(),
            nn.Conv2d(mid, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

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

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = hw_shape
        feat = nlc_to_nchw(x, (h, w))

        feat = self.act(self.pw1(feat))

        # Depthwise aggregation
        agg = 0
        for dw in self.dw_branches:
            agg = agg + dw(feat)
        if self.use_directional:
            agg = agg + self.dw_h(feat) + self.dw_v(feat)

        gate = self.se(agg)
        out = feat + self.pw2(gate * agg)

        return nchw_to_nlc(out)


class FullCrossAttention(BaseModule):
    """Standard cross-attention (Q from x, K/V from kv)."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert dim_q % num_heads == 0, f"dim_q({dim_q}) must be divisible by num_heads({num_heads})."

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
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

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n_q, c_q = x.shape
        b2, n_k, c_k = kv.shape
        assert b == b2, "Batch size mismatch between query and kv."

        q = self.q(x).reshape(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B,h,Nq,dh
        kv = self.kv(kv).reshape(b, n_k, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B,h,Nk,dh

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B,h,Nq,Nk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n_q, c_q)  # B,Nq,C
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LowRankCrossAttention(BaseModule):
    """Rank-r compressed cross-attention.

    Attention logits are computed on a low-rank subspace (rank r) per head:
        A = Softmax( (Q_r K_r^T) / sqrt(r) )
    where Q_r = W_qr x, K_r = W_kr kv.
    V uses full head_dim for output quality.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        rank: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert dim_q % num_heads == 0, f"dim_q({dim_q}) must be divisible by num_heads({num_heads})."
        head_dim = dim_q // num_heads
        assert 1 <= rank <= head_dim, f"rank({rank}) must be in [1, head_dim({head_dim})]."

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rank = rank
        self.scale = rank ** -0.5

        self.qr = nn.Linear(dim_q, num_heads * rank, bias=qkv_bias)
        self.kr = nn.Linear(dim_kv, num_heads * rank, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
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

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n_q, c_q = x.shape
        b2, n_k, _ = kv.shape
        assert b == b2, "Batch size mismatch between query and kv."

        q_r = self.qr(x).reshape(b, n_q, self.num_heads, self.rank).permute(0, 2, 1, 3)  # B,h,Nq,r
        k_r = self.kr(kv).reshape(b, n_k, self.num_heads, self.rank).permute(0, 2, 1, 3)  # B,h,Nk,r

        attn = (q_r @ k_r.transpose(-2, -1)) * self.scale  # B,h,Nq,Nk
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = self.v(kv).reshape(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # B,h,Nk,dh
        out = (attn @ v).transpose(1, 2).reshape(b, n_q, c_q)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DualRouteCrossAttention(BaseModule):
    """Coordinate-Aligned Dual-Route Cross Attention (CADRA).

    This module replaces vanilla cross-attention with a **bi-branch** design:

    1) Local route (spatially aligned):
       Each query token attends only to a small k×k window on the low-resolution KV grid.
       This keeps boundary/detail information sharp and avoids irrelevant global mixing.

    2) Global route (context condensed):
       Each query token attends to a compact set of global tokens obtained by adaptive pooling
       on the KV grid (and optionally extra strip tokens if provided by the KV builder).

    To further improve efficiency, similarity is computed in a low-dimensional embedding
    (reduction_dim per head), inspired by Channel-Reduction Attention / Strip Cross-Attention.

    Notes:
        - kv_tokens are assumed to be built on a KV grid of shape (Hk, Wk).
        - If kv_tokens include extra tokens after the grid tokens (e.g., strip tokens),
          they will be treated as additional global tokens when `use_kv_strip=True`.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        reduction_dim: int = 4,
        local_window_size: int = 3,
        global_pool_size: int = 4,
        use_kv_strip: bool = False,
        router_per_head: bool = True,
        router_mlp_ratio: float = 0.25,
        use_rel_pos_bias: bool = True,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert dim_q % num_heads == 0, f"dim_q({dim_q}) must be divisible by num_heads({num_heads})."
        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_q // self.num_heads

        self.reduction_dim = int(reduction_dim)
        if self.reduction_dim <= 0:
            raise ValueError('reduction_dim must be > 0')

        self.scale = self.reduction_dim ** -0.5

        self.local_window_size = int(local_window_size)
        if self.local_window_size > 0 and self.local_window_size % 2 != 1:
            raise ValueError('local_window_size must be odd when enabled (>0).')

        self.global_pool_size = int(global_pool_size)
        self.use_kv_strip = bool(use_kv_strip)

        # Low-dim similarity projections (Q and K)
        self.qr = nn.Linear(self.dim_q, self.num_heads * self.reduction_dim, bias=qkv_bias)
        self.kr = nn.Linear(self.dim_kv, self.num_heads * self.reduction_dim, bias=qkv_bias)

        # Value projection uses full dim_q for output quality
        self.v = nn.Linear(self.dim_kv, self.dim_q, bias=qkv_bias)

        # Local relative position bias (very cheap)
        self.use_rel_pos_bias = bool(use_rel_pos_bias) and (self.local_window_size > 0)
        if self.use_rel_pos_bias:
            k2 = self.local_window_size * self.local_window_size
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.num_heads, k2))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        else:
            self.rel_pos_bias = None

        # Lightweight router: adaptively balance local/global per query (and optionally per head)
        self.router_per_head = bool(router_per_head)
        hidden = max(int(self.dim_q * float(router_mlp_ratio)), 4)
        out_dim = self.num_heads if self.router_per_head else 1
        self.router = nn.Sequential(
            nn.Linear(self.dim_q, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_q, self.dim_q)
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

    @staticmethod
    def _build_aligned_index(
        hw_q: Tuple[int, int],
        hw_k: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build integer indices that map each query location to a KV-grid location.

        The mapping uses center-aligned floor:
            y_k = floor((y_q + 0.5) * Hk / Hq)
            x_k = floor((x_q + 0.5) * Wk / Wq)

        Returns:
            idx: (Hq*Wq,) long tensor in [0, Hk*Wk-1]
        """
        hq, wq = int(hw_q[0]), int(hw_q[1])
        hk, wk = int(hw_k[0]), int(hw_k[1])
        if hq <= 0 or wq <= 0 or hk <= 0 or wk <= 0:
            raise ValueError('Invalid hw shapes for aligned index.')

        ys = (torch.arange(hq, device=device, dtype=torch.float32) + 0.5) * (float(hk) / float(hq))
        xs = (torch.arange(wq, device=device, dtype=torch.float32) + 0.5) * (float(wk) / float(wq))
        ys = ys.floor().clamp(0, hk - 1).to(torch.long)
        xs = xs.floor().clamp(0, wk - 1).to(torch.long)

        idx = (ys[:, None] * wk + xs[None, :]).reshape(-1)
        return idx

    def forward(
        self,
        x: torch.Tensor,  # (B, Nq, Cq)
        kv_tokens: torch.Tensor,  # (B, Nk, Ckv) where Nk >= Hk*Wk
        hw_shape: Tuple[int, int],  # (Hq, Wq)
        kv_hw_shape: Tuple[int, int],  # (Hk, Wk) for the grid tokens
    ) -> torch.Tensor:
        b, n_q, _ = x.shape
        hk, wk = int(kv_hw_shape[0]), int(kv_hw_shape[1])
        n_grid = hk * wk
        if kv_tokens.size(1) < n_grid:
            raise ValueError(
                f'kv_tokens length {kv_tokens.size(1)} is smaller than Hk*Wk={n_grid}. '
                'Please check kv_hw_shape.',
            )

        kv_grid = kv_tokens[:, :n_grid, :]  # (B, HkWk, Ckv)
        kv_extra = kv_tokens[:, n_grid:, :] if kv_tokens.size(1) > n_grid else None

        # ---- Q reduced embedding ----
        q_r = self.qr(x).reshape(b, n_q, self.num_heads, self.reduction_dim).permute(0, 2, 1, 3)  # B,h,Nq,dr

        # ---- Build reduced K and V maps from grid tokens (project once) ----
        k_r = self.kr(kv_grid).reshape(b, n_grid, self.num_heads, self.reduction_dim).permute(0, 2, 3, 1)
        k_r_map = k_r.reshape(b, self.num_heads * self.reduction_dim, hk, wk)  # (B, h*dr, Hk, Wk)

        v = self.v(kv_grid).reshape(b, n_grid, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v_map = v.reshape(b, self.num_heads * self.head_dim, hk, wk)  # (B, h*dh, Hk, Wk)

        # ---- Global tokens (pooled) ----
        k_global = None
        v_global = None
        g = int(self.global_pool_size)
        if g > 0:
            g_eff = min(g, hk, wk)
            if g_eff > 0:
                k_pool = F.adaptive_avg_pool2d(k_r_map, output_size=(g_eff, g_eff))
                v_pool = F.adaptive_avg_pool2d(v_map, output_size=(g_eff, g_eff))

                k_pool = k_pool.view(b, self.num_heads, self.reduction_dim, g_eff * g_eff).permute(0, 1, 3, 2)
                v_pool = v_pool.view(b, self.num_heads, self.head_dim, g_eff * g_eff).permute(0, 1, 3, 2)
                k_global = k_pool  # (B,h,Ng,dr)
                v_global = v_pool  # (B,h,Ng,dh)

        # Optional: append strip/global extra tokens produced by KV builder
        if self.use_kv_strip and kv_extra is not None and kv_extra.numel() > 0:
            n_extra = kv_extra.size(1)
            k_extra = self.kr(kv_extra).reshape(b, n_extra, self.num_heads, self.reduction_dim).permute(0, 2, 1, 3)
            v_extra = self.v(kv_extra).reshape(b, n_extra, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            if k_global is None:
                k_global = k_extra
                v_global = v_extra
            else:
                k_global = torch.cat([k_extra, k_global], dim=2)
                v_global = torch.cat([v_extra, v_global], dim=2)

        # ---- Local tokens (aligned k×k windows) ----
        out_local = None
        if self.local_window_size > 0:
            win = int(self.local_window_size)
            pad = win // 2

            idx = self._build_aligned_index(hw_shape, (hk, wk), device=x.device)  # (Nq,)
            # unfold on projected maps
            k_unf = F.unfold(k_r_map, kernel_size=win, padding=pad)  # (B, h*dr*win^2, HkWk)
            v_unf = F.unfold(v_map, kernel_size=win, padding=pad)  # (B, h*dh*win^2, HkWk)

            # gather windows for each query position
            k_unf = k_unf.transpose(1, 2).contiguous()  # (B, HkWk, h*dr*win^2)
            v_unf = v_unf.transpose(1, 2).contiguous()  # (B, HkWk, h*dh*win^2)

            k_sel = torch.gather(
                k_unf, dim=1, index=idx.view(1, -1, 1).expand(b, -1, k_unf.size(-1)),
            )  # (B, Nq, h*dr*win^2)
            v_sel = torch.gather(
                v_unf, dim=1, index=idx.view(1, -1, 1).expand(b, -1, v_unf.size(-1)),
            )  # (B, Nq, h*dh*win^2)

            # reshape to per-head windows
            k_sel = k_sel.view(b, n_q, self.num_heads * self.reduction_dim, win * win)
            k_sel = k_sel.view(b, n_q, self.num_heads, self.reduction_dim, win * win).permute(0, 2, 1, 4, 3)
            # (B,h,Nq,win^2,dr)

            v_sel = v_sel.view(b, n_q, self.num_heads * self.head_dim, win * win)
            v_sel = v_sel.view(b, n_q, self.num_heads, self.head_dim, win * win).permute(0, 2, 1, 4, 3)
            # (B,h,Nq,win^2,dh)

            logits_local = (q_r.unsqueeze(3) * k_sel).sum(dim=-1) * self.scale  # (B,h,Nq,win^2)
            if self.rel_pos_bias is not None:
                logits_local = logits_local + self.rel_pos_bias.view(1, self.num_heads, 1, win * win)

            attn_local = logits_local.softmax(dim=-1)
            attn_local = self.attn_drop(attn_local)
            out_local = (attn_local.unsqueeze(-1) * v_sel).sum(dim=3)  # (B,h,Nq,dh)

        # ---- Global attention ----
        out_global = None
        if k_global is not None and v_global is not None:
            logits_global = (q_r @ k_global.transpose(-2, -1)) * self.scale  # (B,h,Nq,Ng)
            attn_global = logits_global.softmax(dim=-1)
            attn_global = self.attn_drop(attn_global)
            out_global = attn_global @ v_global  # (B,h,Nq,dh)

        if out_local is None and out_global is None:
            raise RuntimeError('Both local and global branches are disabled in DualRouteCrossAttention.')

        if out_local is None:
            out = out_global
        elif out_global is None:
            out = out_local
        else:
            gate = torch.sigmoid(self.router(x))  # (B,Nq,h) or (B,Nq,1)
            if self.router_per_head:
                gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B,h,Nq,1)
            else:
                gate = gate.unsqueeze(1).unsqueeze(-1)  # (B,1,Nq,1)
            out = gate * out_local + (1.0 - gate) * out_global  # (B,h,Nq,dh)

        out = out.transpose(1, 2).reshape(b, n_q, self.dim_q)  # (B,Nq,Cq)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class StageGatedKVFusion(BaseModule):
    """Pool & (optional) stage-gate multi-scale features to build the KV memory.

    Expected input feature order: [s4, s3, s2, s1]  (deep -> shallow).
    pool_ratio should also follow [s4, s3, s2, s1], e.g. [1,2,4,8] if KV target is s4.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        query_channels: Optional[int] = None,
        with_gate: bool = True,
        gate_dim: int = 32,
        with_strip_tokens: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert len(in_channels) == 4, "This fusion module assumes 4 stages."
        assert len(pool_ratio) == 4, "pool_ratio must have length 4 (s4->s1)."
        self.in_channels = list(map(int, in_channels))
        self.pool_ratio = list(map(int, pool_ratio))
        self.with_gate = bool(with_gate)
        self.with_strip_tokens = bool(with_strip_tokens)

        # Only build pooling/proj modules for stages with ratio>1 (exactly following CatKey in the official impl)
        self._pool_inds = [i for i, r in enumerate(self.pool_ratio) if r > 1]
        self.pool_list = ModuleList(
            [
                nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in self._pool_inds
            ],
        )
        self.sr_list = ModuleList(
            [
                nn.Conv2d(self.in_channels[i], self.in_channels[i], kernel_size=1, stride=1, bias=True)
                for i in self._pool_inds
            ],
        )

        # Stage-gate (optional). Gating is conditioned on query's global descriptor.
        # Each stage gets a scalar weight, normalized by softmax.
        if self.with_gate:
            assert query_channels is not None, "query_channels must be set when with_gate=True."
            # Tiny MLP with 1x1 convs to produce 4-way softmax per spatial location.
            self.spatial_gate = Sequential(
                nn.Conv2d(query_channels + self.out_channels, gate_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv2d(gate_dim, 4, kernel_size=1, bias=True),
            )

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

    @property
    def out_channels(self) -> int:
        return int(sum(self.in_channels))

    def forward(
        self,
        feats_s4_to_s1: Sequence[torch.Tensor],
        query_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert len(feats_s4_to_s1) == 4, "feats_s4_to_s1 must be a list/tuple of 4 tensors."
        out_list: List[torch.Tensor] = []

        cnt = 0
        for i in range(4):
            x_i = feats_s4_to_s1[i]
            if self.pool_ratio[i] > 1:
                x_i = self.sr_list[cnt](self.pool_list[cnt](x_i))
                cnt += 1
            # else keep identity (no extra conv) just like official CatKey
            out_list.append(x_i)

        if self.with_gate:
            assert query_feat is not None, "query_feat is required when with_gate=True."
            # Pool query to KV spatial size
            h_k, w_k = out_list[0].shape[-2:]
            q_pool = F.adaptive_avg_pool2d(query_feat, (h_k, w_k))
            gate_in = torch.cat([q_pool] + out_list, dim=1)
            gate = torch.softmax(self.spatial_gate(gate_in), dim=1)  # (B,4,Hk,Wk)
            for j in range(4):
                out_list[j] = out_list[j] * gate[:, j:j + 1]

        mem_map = torch.cat(out_list, dim=1)  # (B, C_total, Hk, Wk)

        # Tokenization: grid tokens always, optional strip tokens (Hk + Wk)
        grid = nchw_to_nlc(mem_map)  # (B, Hk*Wk, C_total)
        if not self.with_strip_tokens:
            return grid

        # horizontal strip: avg over width -> (B, C, Hk) -> (B, Hk, C)
        h_strip = mem_map.mean(dim=3).permute(0, 2, 1).contiguous()
        # vertical strip: avg over height -> (B, C, Wk) -> (B, Wk, C)
        w_strip = mem_map.mean(dim=2).permute(0, 2, 1).contiguous()

        kv_tokens = torch.cat([grid, h_strip, w_strip], dim=1)
        return kv_tokens


class GHRBlock(BaseModule):
    """Decoder block with pluggable attention + optional LPU & FFN.

    Supported attention types:
        - 'auto': choose full/low-rank by attn_rank (keeps backward compatibility).
        - 'full': standard cross-attention (baseline).
        - 'low_rank': rank-r compressed cross-attention (ablation).
        - 'dual_route': Coordinate-Aligned Dual-Route Cross Attention (CADRA, ours).
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attn_type: str = 'auto',
        attn_rank: Optional[int] = None,
        # ---- dual-route attention params ----
        dual_route_reduction_dim: int = 4,
        dual_route_local_window: int = 3,
        dual_route_global_pool: int = 4,
        dual_route_use_kv_strip: bool = False,
        dual_route_router_per_head: bool = True,
        dual_route_router_mlp_ratio: float = 0.25,
        dual_route_use_rel_pos_bias: bool = True,
        # -----------------------------------
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        with_lpu: bool = True,
        lpu_dilations: Sequence[int] = (1, 2),
        lpu_use_directional: bool = True,
        lpu_directional_kernel: int = 7,
        ffn_type: str = 'gated',  # 'mlp' or 'gated'
        mlp_ratio: float = 4.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.attn_type = str(attn_type).lower()
        assert self.attn_type in ['auto', 'full', 'low_rank', 'dual_route'], (
            "attn_type must be one of ['auto','full','low_rank','dual_route']."
        )

        head_dim = dim_q // num_heads

        if self.attn_type == 'dual_route':
            self.attn = DualRouteCrossAttention(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                reduction_dim=int(dual_route_reduction_dim),
                local_window_size=int(dual_route_local_window),
                global_pool_size=int(dual_route_global_pool),
                use_kv_strip=bool(dual_route_use_kv_strip),
                router_per_head=bool(dual_route_router_per_head),
                router_mlp_ratio=float(dual_route_router_mlp_ratio),
                use_rel_pos_bias=bool(dual_route_use_rel_pos_bias),
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            # Attention implementation ablation:
            # - 'full': always full attention
            # - 'low_rank': use LowRankCrossAttention when 1 <= rank < head_dim, otherwise fallback to full
            if self.attn_type == 'full':
                self.attn = FullCrossAttention(
                    dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads,
                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
                )
            else:
                if attn_rank is None or attn_rank <= 0 or attn_rank >= head_dim:
                    self.attn = FullCrossAttention(
                        dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads,
                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
                    )
                else:
                    self.attn = LowRankCrossAttention(
                        dim_q=dim_q, dim_kv=dim_kv, num_heads=num_heads, rank=int(attn_rank),
                        qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,
                    )

        self.drop_path = DropPath(float(drop_path))

        # Optional Local Perception Unit (ablation)
        self.with_lpu = bool(with_lpu)
        if self.with_lpu:
            self.norm_lpu = nn.LayerNorm(dim_q)
            self.lpu = DirectionalDilatedLPU(
                dim=dim_q,
                dilations=lpu_dilations,
                use_directional=lpu_use_directional,
                directional_kernel=lpu_directional_kernel,
            )

        # FFN (ablation)
        self.norm_ffn = nn.LayerNorm(dim_q)
        ffn_type = str(ffn_type).lower()
        assert ffn_type in ['mlp', 'gated']
        self.ffn_type = ffn_type
        if self.ffn_type == 'mlp':
            self.ffn = MlpDWConv(dim=dim_q, mlp_ratio=mlp_ratio, drop=drop)
        else:
            self.ffn = GatedDWConvFFN(dim=dim_q, mlp_ratio=mlp_ratio, drop=drop)

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

    def forward(
        self,
        x: torch.Tensor,  # (B, Nq, Cq)
        kv_tokens: torch.Tensor,  # (B, Nk, Ckv)
        hw_shape: Tuple[int, int],  # (Hq,Wq)
        kv_hw_shape: Optional[Tuple[int, int]] = None,  # (Hk,Wk) if needed
    ) -> torch.Tensor:
        if kv_hw_shape is None:
            kv_hw_shape = hw_shape

        if self.attn_type == 'dual_route':
            x = x + self.drop_path(
                self.attn(self.norm_q(x), self.norm_kv(kv_tokens), hw_shape=hw_shape, kv_hw_shape=kv_hw_shape),
            )
        else:
            x = x + self.drop_path(self.attn(self.norm_q(x), self.norm_kv(kv_tokens)))

        if self.with_lpu:
            x = x + self.drop_path(self.lpu(self.norm_lpu(x), hw_shape))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x), hw_shape))
        return x


@MODELS.register_module()
class GHRMixFormerHead(BaseDecodeHead):
    """Ablation-friendly enhanced U-MixFormer decoder head.

    Stage order conventions in this implementation:
        - Inputs from backbone are (c1, c2, c3, c4) = (1/4, 1/8, 1/16, 1/32).
        - Any *list* stage-wise argument is ordered as [s4, s3, s2, s1] (deep -> shallow).
    """

    def __init__(
        self,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),  # s4->s1, target KV resolution = s4
        num_heads: Sequence[int] = (8, 5, 2, 1),  # s4->s1
        attn_rank: Union[int, Sequence[int], None] = (4, 4, 2, 1),  # s4->s1, None/full -> baseline full attn
        # ---- Core attention type (token mixer) ----
        attn_type: Union[str, Sequence[str]] = 'auto',  # 'auto' | 'full' | 'low_rank' | 'dual_route'
        dual_route_reduction_dim: Union[int, Sequence[int]] = (4, 4, 2, 1),  # s4->s1
        dual_route_local_window: Union[int, Sequence[int]] = (0, 3, 3, 3),  # s4->s1
        dual_route_global_pool: Union[int, Sequence[int]] = (4, 4, 4, 4),  # s4->s1
        dual_route_use_kv_strip: bool = False,
        dual_route_router_per_head: bool = True,
        dual_route_router_mlp_ratio: float = 0.25,
        dual_route_use_rel_pos_bias: bool = True,

        with_strip_tokens: bool = True,
        kv_gate: bool = True,
        kv_gate_dim: Union[int, Sequence[int]] = 32,

        with_lpu: Union[bool, Sequence[bool]] = True,  # global or stage-wise (s4->s1)
        lpu_dilations: Sequence[int] = (1, 2),
        lpu_use_directional: bool = True,
        lpu_directional_kernel: int = 7,

        ffn_type: Union[str, Sequence[str]] = 'gated',  # global or stage-wise (s4->s1)
        mlp_ratio: Union[float, Sequence[float]] = 4.0,  # global or stage-wise (s4->s1)

        # ---- ELAR ----
        with_elar: bool = False,
        elar_kernel_size: int = 5,
        elar_guidance_channels: int = 64,
        elar_theta_channels: int = 16,
        elar_use_prob: bool = True,
        elar_prob_channels: int = 16,
        elar_detach_prob: bool = True,
        elar_num_iters: int = 1,
        elar_residual: bool = True,
        elar_residual_weight: float = 1.0,

        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.1,

        # ---- fuse layer norm config ----
        fuse_norm_cfg: Optional[dict] = dict(type='SyncBN', requires_grad=True),

        # ---- BaseDecodeHead args (explicit for clarity) ----
        in_channels: Sequence[int] = (32, 64, 160, 256),
        channels: int = 256,
        num_classes: int = 150,
        in_index: Sequence[int] = (0, 1, 2, 3),
        input_transform: str = 'multiple_select',
        dropout_ratio: float = 0.1,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: dict = dict(type='ReLU'),
        loss_decode: dict = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ignore_index: int = 255,
        align_corners: bool = False,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            loss_decode=loss_decode,
            ignore_index=ignore_index,
            align_corners=align_corners,
            init_cfg=init_cfg,
        )
        # Stage channels (note: in_channels is [c1,c2,c3,c4] in mmseg convention)
        c1, c2, c3, c4 = list(self.in_channels)
        self._stage_channels_s4_to_s1 = [c4, c3, c2, c1]
        tot_channels = sum(self._stage_channels_s4_to_s1)

        # Normalize list-like configs to length 4 (s4->s1)
        def _norm_stage_list(v, name: str):
            if isinstance(v, (list, tuple)):
                assert len(v) == 4, f'{name} must have length 4 (s4->s1) when list/tuple.'
                return list(v)
            return [v, v, v, v]

        num_heads = _norm_stage_list(num_heads, 'num_heads')
        attn_rank = _norm_stage_list(attn_rank, 'attn_rank')
        attn_type = _norm_stage_list(attn_type, 'attn_type')
        dual_route_reduction_dim = _norm_stage_list(dual_route_reduction_dim, 'dual_route_reduction_dim')
        dual_route_local_window = _norm_stage_list(dual_route_local_window, 'dual_route_local_window')
        dual_route_global_pool = _norm_stage_list(dual_route_global_pool, 'dual_route_global_pool')
        kv_gate_dim = _norm_stage_list(kv_gate_dim, 'kv_gate_dim')
        with_lpu = _norm_stage_list(with_lpu, 'with_lpu')
        ffn_type = _norm_stage_list(ffn_type, 'ffn_type')
        mlp_ratio = _norm_stage_list(mlp_ratio, 'mlp_ratio')
        drop_path_rate = _norm_stage_list(drop_path_rate, 'drop_path_rate')

        # ---- KV fusion per stage (query_channels differs) ----
        self.kv_fusion = ModuleList(
            [
                StageGatedKVFusion(
                    in_channels=self._stage_channels_s4_to_s1,
                    pool_ratio=pool_ratio,
                    query_channels=self._stage_channels_s4_to_s1[i],
                    with_gate=kv_gate,
                    gate_dim=kv_gate_dim[i],
                    with_strip_tokens=with_strip_tokens,
                ) for i in range(4)
            ],
        )

        # ---- Decoder blocks (s4->s1) ----
        dims_q = self._stage_channels_s4_to_s1
        self.blocks = ModuleList()
        for i in range(4):
            self.blocks.append(
                GHRBlock(
                    dim_q=dims_q[i],
                    dim_kv=tot_channels,
                    num_heads=int(num_heads[i]),
                    attn_type=str(attn_type[i]),
                    dual_route_reduction_dim=int(dual_route_reduction_dim[i]),
                    dual_route_local_window=int(dual_route_local_window[i]),
                    dual_route_global_pool=int(dual_route_global_pool[i]),
                    dual_route_use_kv_strip=bool(dual_route_use_kv_strip),
                    dual_route_router_per_head=bool(dual_route_router_per_head),
                    dual_route_router_mlp_ratio=float(dual_route_router_mlp_ratio),
                    dual_route_use_rel_pos_bias=bool(dual_route_use_rel_pos_bias),
                    attn_rank=None if attn_rank[i] is None else int(attn_rank[i]),
                    qkv_bias=True,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop=drop,
                    drop_path=float(drop_path_rate[i]),
                    with_lpu=bool(with_lpu[i]),
                    lpu_dilations=lpu_dilations,
                    lpu_use_directional=lpu_use_directional,
                    lpu_directional_kernel=lpu_directional_kernel,
                    ffn_type=str(ffn_type[i]),
                    mlp_ratio=float(mlp_ratio[i]),
                ),
            )

        # ---- Fusion conv (concat D4..D1 -> channels) ----
        if ConvModule is None:
            self.linear_fuse = nn.Conv2d(tot_channels, channels, kernel_size=1, bias=True)
        else:
            self.linear_fuse = ConvModule(
                in_channels=tot_channels,
                out_channels=channels,
                kernel_size=1,
                norm_cfg=fuse_norm_cfg,
            )

        # ---- ELAR Refine ----
        self.with_elar = with_elar
        if self.with_elar:
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

    def forward(self, inputs):
        # inputs: list of backbone feature maps in order [c1,c2,c3,c4]
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # Stage s4
        kv4 = self.kv_fusion[0]([c4, c3, c2, c1], query_feat=c4)  # tokens
        q4 = nchw_to_nlc(c4)
        d4 = self.blocks[0](q4, kv4, (h4, w4), kv_hw_shape=(h4, w4))
        d4 = nlc_to_nchw(d4, (h4, w4))

        # Stage s3
        kv3 = self.kv_fusion[1]([d4, c3, c2, c1], query_feat=c3)
        q3 = nchw_to_nlc(c3)
        d3 = self.blocks[1](q3, kv3, (h3, w3), kv_hw_shape=(h4, w4))
        d3 = nlc_to_nchw(d3, (h3, w3))

        # Stage s2
        kv2 = self.kv_fusion[2]([d4, d3, c2, c1], query_feat=c2)
        q2 = nchw_to_nlc(c2)
        d2 = self.blocks[2](q2, kv2, (h2, w2), kv_hw_shape=(h4, w4))
        d2 = nlc_to_nchw(d2, (h2, w2))

        # Stage s1
        kv1 = self.kv_fusion[3]([d4, d3, d2, c1], query_feat=c1)
        q1 = nchw_to_nlc(c1)
        d1 = self.blocks[3](q1, kv1, (h1, w1), kv_hw_shape=(h4, w4))
        d1 = nlc_to_nchw(d1, (h1, w1))

        # Upsample & fuse
        if resize is None:
            d4_up = F.interpolate(d4, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
            d3_up = F.interpolate(d3, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
            d2_up = F.interpolate(d2, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        else:
            d4_up = resize(d4, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
            d3_up = resize(d3, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
            d2_up = resize(d2, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        fused = self.linear_fuse(torch.cat([d4_up, d3_up, d2_up, d1], dim=1))
        fused = self.cls_seg(fused)

        if self.with_elar and self.elar is not None:
            fused = self.elar(fused, feat_high=d1, feat_low=c1)
        return fused
