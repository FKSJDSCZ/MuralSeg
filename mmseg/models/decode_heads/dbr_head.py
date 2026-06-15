# DBR-Decoder: Dual-resolution Boundary-biased Rank-r Decoder Head
# ---------------------------------------------------------------
# This file is designed to be dropped into:
#   mmseg/models/decode_heads/dbr_head.py
# and then imported in mmseg/models/decode_heads/__init__.py

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS

# Support both "inside-mmseg" relative import and "outside-mmseg" absolute import.
try:
    # mmseg/models/decode_heads/dbr_head.py -> mmseg/models/utils/__init__.py
    from ..utils import resize  # type: ignore
    from .decode_head import BaseDecodeHead  # type: ignore
except Exception:  # pragma: no cover
    from mmseg.models.utils import resize  # type: ignore
    from mmseg.models.decode_heads.decode_head import BaseDecodeHead  # type: ignore


def _as_tuple(size: Sequence[int]) -> Tuple[int, int]:
    assert len(size) == 2
    return int(size[0]), int(size[1])


def _resize_to(
    x: Tensor,
    size: Tuple[int, int],
    mode: str = 'bilinear',
    align_corners: bool = False,
) -> Tensor:
    """Resize feature map to target spatial size.

    Following DBR.md:
      - Downsample prefers AvgPool (low-pass) rather than interpolation.
      - Upsample uses bilinear.

    Args:
        x (Tensor): (B, C, H, W)
        size (Tuple[int,int]): (H_t, W_t)
    """
    h, w = x.shape[-2:]
    th, tw = size
    if (h, w) == (th, tw):
        return x

    # Downsample: use adaptive average pooling to exactly match target size.
    if th <= h and tw <= w:
        return F.adaptive_avg_pool2d(x, output_size=size)

    # Upsample (or mixed case): use resize util (wrapper around interpolate).
    return resize(input=x, size=size, mode=mode, align_corners=align_corners)


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW features.

    Equivalent to nn.LayerNorm(C) applied per spatial location.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class BoundaryPrior(BaseModule):
    """Low-frequency structural boundary prior extractor.

    Implements Eq.(E) in DBR.md:
        LP_m(x) = AvgPool(k_m,k_m)(x) ↑
        E = Norm( sum_m || ∇ LP_m(F1) ||_1 )

    Practical implementation notes:
      - We compute gradients on a single-channel structural map (channel-mean)
        for efficiency. AvgPool/mean/gradient are linear, so this is a faithful
        approximation to channel-wise processing while being cheaper.
      - Sobel kernels are fixed buffers (no learnable params).
    """

    def __init__(
        self,
        pool_kernels: Sequence[int] = (3, 7),
        eps: float = 1e-6,
        interpolate_mode: str = 'bilinear',
        align_corners: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.pool_kernels = tuple(int(k) for k in pool_kernels)
        self.eps = float(eps)
        self.interpolate_mode = str(interpolate_mode)
        self.align_corners = bool(align_corners)

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x, persistent=False)
        self.register_buffer('sobel_y', sobel_y, persistent=False)

    @torch.no_grad()
    def _low_pass(self, x: Tensor, k: int) -> Tensor:
        """Downsample by AvgPool(k,k) then upsample back."""
        b, c, h, w = x.shape
        k_eff = int(min(k, h, w))
        if k_eff <= 1:
            return x
        # AvgPool for low-pass + downsample
        y = F.avg_pool2d(x, kernel_size=k_eff, stride=k_eff)
        # Upsample back
        y = resize(
            input=y,
            size=(h, w),
            mode=self.interpolate_mode,
            align_corners=self.align_corners,
        )
        return y

    def forward(self, f1: Tensor) -> Tensor:
        """Compute boundary prior E.

        Args:
            f1: (B, C, H, W), typically the highest-resolution encoder feature.

        Returns:
            E: (B, 1, H, W) normalized to [0, 1].
        """
        # Structural map (B,1,H,W)
        x = f1.mean(dim=1, keepdim=True)

        e = 0.0
        for k in self.pool_kernels:
            lp = self._low_pass(x, k)
            gx = F.conv2d(lp, self.sobel_x, padding=1)
            gy = F.conv2d(lp, self.sobel_y, padding=1)
            e = e + gx.abs() + gy.abs()

        # Min-max normalize per sample -> [0,1]
        e_min = e.amin(dim=(2, 3), keepdim=True)
        e_max = e.amax(dim=(2, 3), keepdim=True)
        e = (e - e_min) / (e_max - e_min + self.eps)
        return e


class RankRBiasCrossAttention(BaseModule):
    """Rank-r cross-attention with additive boundary bias (BB-Attn).

    Implements DBR.md Eqs. (Q,K,V), (B^b), (A^b) and (O^b).

    - Q/K projected to very low rank r per head (r=2/4 suggested).
    - Boundary prior bias:  B_pq = -gamma * (e_q[p] - e_k[q])^2
      added to attention logits.

    Args:
        embed_dims: Query embedding dimension C.
        kv_dims: KV embedding dimension (channel) of mixed feature M^b.
        num_heads: heads h.
        rank_r: low rank r for Q/K per head.
        dim_head: value dimension d per head.
        qkv_bias: whether to use bias in q/k/v projections.
        attn_drop: dropout on attention weights.
        proj_drop: dropout on output projection.
        gamma_init: initial gamma (>0). Stored via softplus to keep gamma positive.
        attn_chunk_size: if set, compute attention by chunking query tokens
            to reduce peak memory. Larger is faster; smaller is safer.
    """

    def __init__(
        self,
        embed_dims: int,
        kv_dims: int,
        num_heads: int = 8,
        rank_r: int = 4,
        dim_head: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gamma_init: float = 1.0,
        attn_chunk_size: Optional[int] = None,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert embed_dims > 0
        assert kv_dims > 0
        assert num_heads > 0
        assert rank_r > 0

        self.embed_dims = int(embed_dims)
        self.kv_dims = int(kv_dims)
        self.num_heads = int(num_heads)
        self.rank_r = int(rank_r)
        self.dim_head = int(dim_head) if dim_head is not None else embed_dims // num_heads
        assert self.dim_head > 0, 'dim_head must be positive.'
        assert embed_dims % num_heads == 0 or dim_head is not None, (
            'If dim_head is None, embed_dims must be divisible by num_heads.'
        )

        self.scale = self.rank_r ** -0.5
        self.attn_chunk_size = attn_chunk_size

        self.norm_q = LayerNorm2d(self.embed_dims)
        self.norm_kv = LayerNorm2d(self.kv_dims)

        self.q_proj = nn.Conv2d(
            self.embed_dims, self.num_heads * self.rank_r, kernel_size=1, bias=qkv_bias,
        )
        self.k_proj = nn.Conv2d(
            self.kv_dims, self.num_heads * self.rank_r, kernel_size=1, bias=qkv_bias,
        )
        self.v_proj = nn.Conv2d(
            self.kv_dims, self.num_heads * self.dim_head, kernel_size=1, bias=qkv_bias,
        )
        self.out_proj = nn.Conv2d(
            self.num_heads * self.dim_head, self.embed_dims, kernel_size=1, bias=True,
        )

        self.attn_drop = nn.Dropout(float(attn_drop))
        self.proj_drop = nn.Dropout(float(proj_drop))

        # Positive gamma via softplus parameterization.
        gamma_init = float(gamma_init)
        gamma_init = max(gamma_init, 0.0)
        inv_softplus = math.log(math.exp(gamma_init) - 1.0) if gamma_init > 0 else -20.0
        self._gamma_param = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

    def _gamma(self) -> Tensor:
        return F.softplus(self._gamma_param)

    def forward(self, query: Tensor, kv: Tensor, e_q: Tensor, e_k: Tensor) -> Tensor:
        """Forward.

        Args:
            query: (B, C, Hq, Wq)
            kv:    (B, C_kv, Hk, Wk)
            e_q: boundary prior at query resolution, (B,1,Hq,Wq) or (B,Nq)
            e_k: boundary prior at kv resolution,    (B,1,Hk,Wk) or (B,Nk)

        Returns:
            out: (B, C, Hq, Wq)
        """
        b, _, hq, wq = query.shape
        _, _, hk, wk = kv.shape
        n_q = hq * wq
        n_k = hk * wk

        # Projections (NCHW -> heads)
        q = self.q_proj(self.norm_q(query))  # (B, h*r, Hq, Wq)
        k = self.k_proj(self.norm_kv(kv))  # (B, h*r, Hk, Wk)
        v = self.v_proj(self.norm_kv(kv))  # (B, h*d, Hk, Wk)

        q = q.view(b, self.num_heads, self.rank_r, n_q).permute(0, 1, 3, 2)  # B,h,Nq,r
        k = k.view(b, self.num_heads, self.rank_r, n_k).permute(0, 1, 3, 2)  # B,h,Nk,r
        v = v.view(b, self.num_heads, self.dim_head, n_k).permute(0, 1, 3, 2)  # B,h,Nk,d

        # Boundary prior vectors
        if e_q.dim() == 4:
            e_q = e_q.flatten(2).squeeze(1)  # (B, Nq)
        if e_k.dim() == 4:
            e_k = e_k.flatten(2).squeeze(1)  # (B, Nk)
        assert e_q.shape == (b, n_q), f'Expect e_q shape {(b, n_q)}, got {tuple(e_q.shape)}'
        assert e_k.shape == (b, n_k), f'Expect e_k shape {(b, n_k)}, got {tuple(e_k.shape)}'

        gamma = self._gamma()

        # Attention (optionally chunked along query tokens)
        if self.attn_chunk_size is None or self.attn_chunk_size >= n_q:
            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B,h,Nq,Nk
            bias = -gamma * (e_q.unsqueeze(-1) - e_k.unsqueeze(-2)).pow(2)  # B,Nq,Nk
            logits = logits + bias.unsqueeze(1)
            attn = logits.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = torch.matmul(attn, v)  # B,h,Nq,d
        else:
            cs = int(self.attn_chunk_size)
            outs: List[Tensor] = []
            k_t = k.transpose(-2, -1)  # B,h,r,Nk
            e_k_ = e_k  # (B,Nk)
            for start in range(0, n_q, cs):
                end = min(start + cs, n_q)
                q_c = q[:, :, start:end, :]  # B,h,M,r
                logits = torch.matmul(q_c, k_t) * self.scale  # B,h,M,Nk
                e_q_c = e_q[:, start:end]  # (B,M)
                bias = -gamma * (e_q_c.unsqueeze(-1) - e_k_.unsqueeze(-2)).pow(2)  # B,M,Nk
                logits = logits + bias.unsqueeze(1)
                attn = logits.softmax(dim=-1)
                attn = self.attn_drop(attn)
                outs.append(torch.matmul(attn, v))  # B,h,M,d
            out = torch.cat(outs, dim=2)  # B,h,Nq,d

        out = out.permute(0, 1, 3, 2).reshape(b, self.num_heads * self.dim_head, hq, wq)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


class GateFreeMultiDilatedLocalRefine(BaseModule):
    """GM-LR: gate-free multi-dilated local refinement.

    DBR.md Eq.:
      GM-LR(Z) = Z + PWConv( sum_t DWConv_{d_t}(Z) )
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 3),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        assert len(self.dilations) > 0, 'dilations must be non-empty.'

        self.dw_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=d,
                    dilation=d,
                    groups=self.channels,
                    bias=False,
                )
                for d in self.dilations
            ],
        )
        self.pw = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        s = self.dw_convs[0](x)
        for conv in self.dw_convs[1:]:
            s = s + conv(x)
        return x + self.pw(s)


class MixFFN2d(BaseModule):
    """A lightweight conv-FFN (MixFFN-style) operating on NCHW."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        act_cfg: dict = dict(type='GELU'),
        ffn_drop: float = 0.0,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = int(embed_dims)
        self.feedforward_channels = int(feedforward_channels)

        self.fc1 = nn.Conv2d(self.embed_dims, self.feedforward_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            self.feedforward_channels,
            self.feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.feedforward_channels,
            bias=True,
        )
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(self.feedforward_channels, self.embed_dims, kernel_size=1, bias=True)
        self.drop = nn.Dropout(float(ffn_drop))

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DBRCLB(BaseModule):
    """DBR Cross-Layer Block (DBR-CLB).

    Stage block (DBR.md):
      - Dual-resolution rank-r cross-attention with boundary bias (two branches)
      - Fuse (concat + linear) + residual
      - GM-LR (multi-dilated DWConv sum + PWConv) + residual
      - LN + FFN + residual
    """

    def __init__(
        self,
        channels: int,
        kv_channels: int,
        num_heads: int = 8,
        rank_r: int = 4,
        dim_head: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gamma_init: float = 1.0,
        attn_chunk_size: Optional[int] = None,
        dilations: Sequence[int] = (1, 2, 3),
        ffn_ratio: float = 4.0,
        ffn_drop: float = 0.0,
        act_cfg: dict = dict(type='GELU'),
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.channels = int(channels)
        self.kv_channels = int(kv_channels)

        self.attn_s = RankRBiasCrossAttention(
            embed_dims=self.channels,
            kv_dims=self.kv_channels,
            num_heads=num_heads,
            rank_r=rank_r,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            gamma_init=gamma_init,
            attn_chunk_size=attn_chunk_size,
        )
        self.attn_d = RankRBiasCrossAttention(
            embed_dims=self.channels,
            kv_dims=self.kv_channels,
            num_heads=num_heads,
            rank_r=rank_r,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            gamma_init=gamma_init,
            attn_chunk_size=attn_chunk_size,
        )

        # Fuse two branches without gating: concat + 1x1 linear
        self.fuse = nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, bias=True)

        self.local_refine = GateFreeMultiDilatedLocalRefine(
            channels=self.channels, dilations=dilations,
        )

        self.norm_ffn = LayerNorm2d(self.channels)
        self.ffn = MixFFN2d(
            embed_dims=self.channels,
            feedforward_channels=int(self.channels * float(ffn_ratio)),
            act_cfg=act_cfg,
            ffn_drop=ffn_drop,
        )

    def forward(
        self,
        query: Tensor,
        kv_s: Tensor,
        kv_d: Tensor,
        e_q: Tensor,
        e_s: Tensor,
        e_d: Tensor,
    ) -> Tensor:
        o_s = self.attn_s(query, kv_s, e_q, e_s)
        o_d = self.attn_d(query, kv_d, e_q, e_d)

        z = self.fuse(torch.cat([o_s, o_d], dim=1)) + query
        z = self.local_refine(z)

        out = self.ffn(self.norm_ffn(z)) + z
        return out


@MODELS.register_module()
class DBRHead(BaseDecodeHead):
    """DBR-Decoder head for semantic segmentation (mmseg).

    High-level flow (DBR.md):
      - Project multi-level features to common channels.
      - Extract low-frequency structural boundary prior E from F1.
      - Decode stages from deep->shallow with DBR-CLB:
          Stage4: Q=F4, KV_S@Res(F4), KV_D@min(Res(F3),Res(F4))
          Stage3: Q=F3, KV_S@Res(F4), KV_D@Res(F3)
          Stage2: Q=F2, KV_S@Res(F4), KV_D@Res(F3)
          Stage1: Q=F1, KV_S@Res(F4), KV_D@Res(F3)
      - Upsample D1..D4 to Res(F1), concat, fuse, cls_seg.

    Notes on efficiency:
      - KV resolutions are fixed to coarse (F4) and mid (F3).
      - Q/K are low-rank r per head (r small).
      - Optional attention chunking reduces peak memory.
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        num_heads: int = 8,
        rank_r: int = 4,
        dim_head: Optional[int] = None,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_ratio: float = 4.0,
        ffn_drop: float = 0.0,
        dilations: Sequence[int] = (1, 2, 3),
        boundary_pool_kernels: Sequence[int] = (3, 7),
        boundary_eps: float = 1e-6,
        gamma_init: float = 1.0,
        attn_chunk_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Always use multiple_select for 4-stage features (like SegformerHead).
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = str(interpolate_mode)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index), 'in_channels and in_index must have same length.'
        assert num_inputs == 4, 'DBRHead expects 4-stage features (F1..F4).'

        # Project encoder features to common channels (phi_R's Linear).
        self.projs = nn.ModuleList()
        for i in range(num_inputs):
            self.projs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=None,  # keep it "linear" as phi_R in DBR.md
                ),
            )

        # Boundary prior extracted from the highest-resolution stage (F1).
        self.boundary_prior = BoundaryPrior(
            pool_kernels=boundary_pool_kernels,
            eps=boundary_eps,
            interpolate_mode=self.interpolate_mode,
            align_corners=self.align_corners,
        )

        kv_channels = self.channels * 4  # concat of 4 aligned features

        # Stage blocks, ordered as [stage1, stage2, stage3, stage4] by index 0..3.
        self.stages = nn.ModuleList(
            [
                DBRCLB(
                    channels=self.channels,
                    kv_channels=kv_channels,
                    num_heads=num_heads,
                    rank_r=rank_r,
                    dim_head=dim_head,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    gamma_init=gamma_init,
                    attn_chunk_size=attn_chunk_size,
                    dilations=dilations,
                    ffn_ratio=ffn_ratio,
                    ffn_drop=ffn_drop,
                    act_cfg=dict(type='GELU'),
                )
                for _ in range(4)
            ],
        )

        # Final fusion (MLP-like head): concat upsampled D1..D4 then 1x1 conv.
        self.fusion_conv = ConvModule(
            in_channels=self.channels * 4,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _build_mixer(
        self,
        feats: List[Tensor],
        dec_outs: List[Optional[Tensor]],
        stage_idx: int,
        target_size: Tuple[int, int],
    ) -> Tensor:
        """Build M_i^b by concatenating 4 aligned features.

        Order follows DBR.md:
          Cat_ch( {F_1..F_i} U {D_{i+1}..D_4} ), aligned to target_size.
        """
        assert len(feats) == 4
        assert len(dec_outs) == 4
        assert 0 <= stage_idx <= 3

        parts: List[Tensor] = []

        # Encoder features F1..F_i (index 0..stage_idx)
        for j in range(0, stage_idx + 1):
            parts.append(
                _resize_to(
                    feats[j],
                    target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                ),
            )

        # Decoder outputs D_{i+1}..D_4 (index stage_idx+1..3)
        for k in range(stage_idx + 1, 4):
            dk = dec_outs[k]
            if dk is None:
                raise RuntimeError(
                    f'Decoder output D{k + 1} is None when building mixer for stage {stage_idx + 1}. '
                    'Please check decode order (should be deep->shallow).',
                )
            parts.append(
                _resize_to(
                    dk,
                    target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners,
                ),
            )

        if len(parts) != 4:
            raise AssertionError(f'Expected 4 parts for mixer at stage {stage_idx + 1}, got {len(parts)}')

        return torch.cat(parts, dim=1)  # (B, 4C, H_t, W_t)

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tensor:
        # Receive 4-stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs_list = self._transform_inputs(inputs)
        assert isinstance(inputs_list, list) and len(inputs_list) == 4

        # Project to common channels.
        feats = [proj(x) for proj, x in zip(self.projs, inputs_list)]  # F1..F4 in C=channels
        f1, f2, f3, f4 = feats

        # Boundary prior E from F1 (highest-res).
        e = self.boundary_prior(f1)  # (B,1,H1,W1)

        # Fixed KV resolutions (DBR.md):
        rs_size = _as_tuple(f4.shape[2:])  # Res(F4)
        rd_base_size = _as_tuple(f3.shape[2:])  # Res(F3)

        dec_outs: List[Optional[Tensor]] = [None, None, None, None]

        # Decode deep -> shallow: stage 4 -> 1 (idx 3 -> 0)
        for i in (3, 2, 1, 0):
            query = feats[i]
            q_size = _as_tuple(query.shape[2:])
            rd_size = rs_size if i == 3 else rd_base_size  # min(Res(F3),Res(F4)) for stage4

            kv_s = self._build_mixer(feats, dec_outs, i, target_size=rs_size)
            kv_d = self._build_mixer(feats, dec_outs, i, target_size=rd_size)

            e_q = _resize_to(e, q_size, mode=self.interpolate_mode, align_corners=self.align_corners)
            e_s = _resize_to(e, rs_size, mode=self.interpolate_mode, align_corners=self.align_corners)
            e_d = _resize_to(e, rd_size, mode=self.interpolate_mode, align_corners=self.align_corners)

            dec_outs[i] = self.stages[i](query, kv_s, kv_d, e_q, e_s, e_d)

        # Fuse outputs at the finest resolution (D1).
        d1 = dec_outs[0]
        d2 = dec_outs[1]
        d3 = dec_outs[2]
        d4 = dec_outs[3]
        assert d1 is not None and d2 is not None and d3 is not None and d4 is not None

        out_size = _as_tuple(d1.shape[2:])
        d2u = _resize_to(d2, out_size, mode=self.interpolate_mode, align_corners=self.align_corners)
        d3u = _resize_to(d3, out_size, mode=self.interpolate_mode, align_corners=self.align_corners)
        d4u = _resize_to(d4, out_size, mode=self.interpolate_mode, align_corners=self.align_corners)

        fused = self.fusion_conv(torch.cat([d1, d2u, d3u, d4u], dim=1))
        seg_logits = self.cls_seg(fused)
        return seg_logits
