# ---------------------------------------------------------------
# This file is a research prototype that extends the official
# U-MixFormer decoder head (APFormerHead2) with a new cross-attention
# module for *scale-routed* local-global fusion.
#
# Key design goals (per user request):
# - Improve Q-K-V interaction for multi-scale fusion (accuracy-first).
# - Avoid hard gating; use soft attention/routing.
# - Provide ablation toggles as *direct __init__ args* (no nested dict).
# - Instantiate optional modules only when enabled.
# - Reuse mmcv/mmengine/mmseg base utilities when available.
#
# NOTE: This file is meant to be dropped into
# `mmseg/models/decode_heads/` and imported by that package.
# ---------------------------------------------------------------

import math
from typing import List, Sequence, Tuple, Optional

import torch
import torch.nn as nn

from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from ..utils import resize
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model.weight_init import trunc_normal_init, constant_init, normal_init
from mmengine.model import BaseModule, ModuleList, Sequential


# -------------------------------------------------------------------------
# Reused building blocks (kept minimal and mmseg-friendly)
# -------------------------------------------------------------------------

class DWConv(BaseModule):
    """Depthwise 3x3 conv on token sequence (B,N,C) <-> (B,C,H,W)."""

    def __init__(self, dim: int):
        super().__init__()
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

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModule):
    """MLP with depthwise conv (as in many segmentation transformers)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
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
    """Pool+1x1 conv per-stage then concat along channel dim.

    pool_ratio and dim are ordered as s4->s1 (deep->shallow).
    """

    def __init__(self, pool_ratio: Sequence[int], dim: Sequence[int]):
        super().__init__()
        self.pool_ratio = list(pool_ratio)
        self.dim = list(dim)
        assert len(self.pool_ratio) == len(self.dim)
        self.sr_list = ModuleList(
            [
                nn.Conv2d(self.dim[i], self.dim[i], kernel_size=1, stride=1)
                for i in range(len(self.pool_ratio))
                if self.pool_ratio[i] > 1
            ],
        )
        self.pool_list = ModuleList(
            [
                nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True)
                for i in range(len(self.pool_ratio))
                if self.pool_ratio[i] > 1
            ],
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

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](feats[i])))
                cnt += 1
            else:
                out_list.append(feats[i])
        return torch.cat(out_list, dim=1)


class CrossAttention(BaseModule):
    """Original cross-attention (kept for ablation)."""

    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        pool_ratio: int = -1,
    ):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."
        self.dim1 = dim1
        self.dim2 = dim2
        self.num_heads = num_heads
        head_dim = dim1 // num_heads
        self.pool_ratio = pool_ratio
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, dim1, bias=qkv_bias)
        self.kv = nn.Linear(dim2, dim1 * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)
        self.proj_drop = nn.Dropout(proj_drop)

        # Note: in the official code this pooling branch is present but unused.
        if self.pool_ratio >= 0:
            self.pool = nn.AvgPool2d(self.pool_ratio, self.pool_ratio)
            self.sr = nn.Conv2d(dim2, dim2, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim2)
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
        """x: (B,Nq,C1), y: (B,Nk,C2)."""
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            # Keep the official behaviour (LN+GELU). The spatial pooling is commented out in official.
            x_ = self.norm(y)
            x_ = self.act(x_)
        else:
            x_ = y

        kv = (
            self.kv(x_)
            .reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# -------------------------------------------------------------------------
# Proposed module: Scale-Routed Decomposed Cross-Attention (SRDCA)
# -------------------------------------------------------------------------

class ScaleRoutedCrossAttention(BaseModule):
    """Scale-Routed Decomposed Cross-Attention.

    Inputs:
        x: query tokens, shape (B, Nq, Cq)
        y: concatenated multi-scale kv tokens, shape (B, Nk, sum(C_s))

    Key idea:
    - Split y into per-scale segments (s4..s1).
    - Compute attention per scale with channel-reduced Q/K (dim = qk_dim).
    - Fuse scales with *soft routing weights* (per token, per head) computed
      from query and per-scale key summary.
    - Optionally enrich values with depthwise conv in 2D (no sigmoid gating).

    Ablations are controlled by init args.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dims: Sequence[int],
        num_heads: int,
        qk_dim: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale_routing: bool = True,
        local_value: bool = True,
        local_kernel_size: int = 3,
        local_dilation: int = 1,
        norm_kv: bool = True,
        act_layer=nn.GELU,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.query_dim = int(query_dim)
        self.kv_dims = [int(d) for d in kv_dims]
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.qk_dim = int(qk_dim) if int(qk_dim) > 0 else self.head_dim
        self.scale = self.qk_dim ** -0.5
        self.scale_routing = bool(scale_routing)
        self.local_value = bool(local_value)
        self.norm_kv = bool(norm_kv)
        self.num_scales = len(self.kv_dims)
        assert self.num_scales > 0

        # Q/K for similarity (decomposed attention): low-dim projection.
        self.q_r = nn.Linear(self.query_dim, self.num_heads * self.qk_dim, bias=qkv_bias)
        self.k_r_list = ModuleList(
            [nn.Linear(d, self.num_heads * self.qk_dim, bias=qkv_bias) for d in self.kv_dims],
        )

        # V projection keeps full capacity (to query_dim) per scale.
        self.v_list = ModuleList([nn.Linear(d, self.query_dim, bias=qkv_bias) for d in self.kv_dims])

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.query_dim, self.query_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional per-scale LN for kv (avoid mixing statistics across scales).
        if self.norm_kv:
            self.kv_norms = ModuleList([nn.LayerNorm(d) for d in self.kv_dims])
        else:
            self.kv_norms = None

        # Optional local enhancement on V (depthwise conv on 2D map).
        if self.local_value:
            padding = (local_kernel_size // 2) * local_dilation
            self.local_dwconvs = ModuleList(
                [
                    nn.Conv2d(
                        d,
                        d,
                        kernel_size=local_kernel_size,
                        padding=padding,
                        dilation=local_dilation,
                        groups=d,
                        bias=True,
                    )
                    for d in self.kv_dims
                ],
            )
            self.local_act = act_layer()
        else:
            self.local_dwconvs = None
            self.local_act = None

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

    def _split_kv(self, y: torch.Tensor) -> List[torch.Tensor]:
        # y: (B,Nk,sumC) -> list[(B,Nk,Cs)]
        return list(torch.split(y, self.kv_dims, dim=-1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, Hk: int, Wk: int) -> torch.Tensor:
        """Forward.

        Args:
            x: (B, Nq, Cq)
            y: (B, Nk, sum(C_s)) concatenated kv tokens (s4..s1)
            Hk, Wk: spatial size for kv grid (needed only if local_value=True)
        """
        B, Nq, _ = x.shape
        # (B,h,Nq,d)
        q_r = self.q_r(x).reshape(B, Nq, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)

        y_list = self._split_kv(y)

        # 1) compute per-scale keys and routing scores
        k_r_list: List[torch.Tensor] = []
        route_scores: List[torch.Tensor] = []
        for s in range(self.num_scales):
            ys = y_list[s]
            if self.kv_norms is not None:
                ys_k = self.kv_norms[s](ys)
            else:
                ys_k = ys
            k_r = self.k_r_list[s](ys_k).reshape(B, -1, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
            k_r_list.append(k_r)
            if self.scale_routing:
                # key summary per head
                k_mean = k_r.mean(dim=2)  # (B,h,d)
                # per-token routing score
                score = (q_r * k_mean.unsqueeze(2)).sum(dim=-1) * self.scale  # (B,h,Nq)
                route_scores.append(score)

        if self.scale_routing:
            # (B,h,Nq,S)
            alpha = torch.stack(route_scores, dim=-1).softmax(dim=-1)
        else:
            alpha = None
            uniform = 1.0 / float(self.num_scales)

        # 2) per-scale attention + weighted sum
        out = 0.0
        for s in range(self.num_scales):
            ys = y_list[s]
            k_r = k_r_list[s]

            # Attention map (B,h,Nq,Nk)
            attn = (q_r @ k_r.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Local enhancement on V (optional, no gating)
            if self.local_dwconvs is not None:
                Cs = ys.shape[-1]
                ys_2d = ys.transpose(1, 2).reshape(B, Cs, Hk, Wk)
                local = self.local_act(self.local_dwconvs[s](ys_2d))
                ys = ys + local.flatten(2).transpose(1, 2)

            if self.kv_norms is not None:
                ys_v = self.kv_norms[s](ys)
            else:
                ys_v = ys

            v = self.v_list[s](ys_v).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            out_s = attn @ v  # (B,h,Nq,head_dim)

            if alpha is not None:
                out = out + out_s * alpha[..., s].unsqueeze(-1)
            else:
                out = out + out_s * uniform

        out = out.transpose(1, 2).reshape(B, Nq, self.query_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixDecodeBlock(BaseModule):
    """A decoder block with selectable attention implementation.

    - attn_type='vanilla': original CrossAttention.
    - attn_type='srdca': ScaleRoutedCrossAttention.
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        kv_dims: Sequence[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale=None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        # --- ablation switches ---
        attn_type: str = "srdca",
        srdca_qk_dim: int = 8,
        srdca_scale_routing: bool = True,
        srdca_local_value: bool = True,
        srdca_local_kernel_size: int = 3,
        srdca_local_dilation: int = 1,
        srdca_norm_kv: bool = True,
    ):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.attn_type = str(attn_type).lower()

        self.norm1 = norm_layer(dim1)
        # For SRDCA we avoid a single LN over concatenated multi-scale channels.
        self.norm2 = nn.Identity() if self.attn_type == "srdca" else norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        if self.attn_type == "srdca":
            self.attn = ScaleRoutedCrossAttention(
                query_dim=dim1,
                kv_dims=kv_dims,
                num_heads=num_heads,
                qk_dim=srdca_qk_dim,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                scale_routing=srdca_scale_routing,
                local_value=srdca_local_value,
                local_kernel_size=srdca_local_kernel_size,
                local_dilation=srdca_local_dilation,
                norm_kv=srdca_norm_kv,
                act_layer=act_layer,
            )
        elif self.attn_type == "vanilla":
            self.attn = CrossAttention(
                dim1=dim1,
                dim2=dim2,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                pool_ratio=-1,
            )
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}. Use 'vanilla' or 'srdca'.")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

    def forward(self, x: torch.Tensor, y: torch.Tensor, Hk: int, Wk: int, Hq: int, Wq: int) -> torch.Tensor:
        x = self.norm1(x)
        y = self.norm2(y)
        if self.attn_type == "srdca":
            x = x + self.drop_path(self.attn(x, y, Hk, Wk))
        else:
            x = x + self.drop_path(self.attn(x, y, Hk, Wk))
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, Hq, Wq))
        return x


# -------------------------------------------------------------------------
# New head: SRDCAHead (drop-in alternative to APFormerHead2)
# -------------------------------------------------------------------------

@MODELS.register_module()
class SRDCAHead(BaseDecodeHead):
    """U-MixFormer style decoder head with SRDCA attention.

    Differences vs official APFormerHead2:
    - Attention module can be swapped via `attn_type`.
    - SRDCA adds: scale routing + channel-reduced similarity + local value enhancement.

    All ablation switches are exposed as *direct init args* (no nested dict).
    Stage-wise list args are ordered as s4->s1 (deep->shallow).
    """

    def __init__(
        self,
        # ---- U-MixFormer / decoder hyper-params ----
        num_heads: Sequence[int] = (8, 5, 2, 1),  # s4->s1
        pool_ratio: Sequence[int] = (1, 2, 4, 8),  # s4->s1
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        # ---- ablation: attention impl ----
        attn_type: str = "srdca",  # 'srdca' or 'vanilla'
        # ---- ablation: SRDCA options ----
        srdca_qk_dim: int = 8,
        srdca_scale_routing: bool = True,
        srdca_local_value: bool = True,
        srdca_local_kernel_size: int = 3,
        srdca_local_dilation: int = 1,
        srdca_norm_kv: bool = True,
        # ---- ablation: keep/skip the last (c1) attention stage ----
        use_c1_attn: bool = True,
        # ---- ablation: share CatKey across stages ----
        share_cat_key: bool = False,
        **kwargs,
    ):
        super().__init__(input_transform="multiple_select", **kwargs)
        self.use_c1_attn = bool(use_c1_attn)

        # in_channels are in order: c1,c2,c3,c4 (shallow->deep)
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = sum(self.in_channels)

        # For SRDCA we need to know per-scale channel splits in y (s4->s1).
        kv_dims = [c4_in, c3_in, c2_in, c1_in]

        # Build decoder blocks (ordered s4->s1)
        assert len(num_heads) == 4, "num_heads must have 4 elements (s4->s1)"
        self.attn_c4 = MixDecodeBlock(
            dim1=c4_in,
            dim2=tot_channels,
            kv_dims=kv_dims,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            attn_type=attn_type,
            srdca_qk_dim=srdca_qk_dim,
            srdca_scale_routing=srdca_scale_routing,
            srdca_local_value=srdca_local_value,
            srdca_local_kernel_size=srdca_local_kernel_size,
            srdca_local_dilation=srdca_local_dilation,
            srdca_norm_kv=srdca_norm_kv,
        )
        self.attn_c3 = MixDecodeBlock(
            dim1=c3_in,
            dim2=tot_channels,
            kv_dims=kv_dims,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            attn_type=attn_type,
            srdca_qk_dim=srdca_qk_dim,
            srdca_scale_routing=srdca_scale_routing,
            srdca_local_value=srdca_local_value,
            srdca_local_kernel_size=srdca_local_kernel_size,
            srdca_local_dilation=srdca_local_dilation,
            srdca_norm_kv=srdca_norm_kv,
        )
        self.attn_c2 = MixDecodeBlock(
            dim1=c2_in,
            dim2=tot_channels,
            kv_dims=kv_dims,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            attn_type=attn_type,
            srdca_qk_dim=srdca_qk_dim,
            srdca_scale_routing=srdca_scale_routing,
            srdca_local_value=srdca_local_value,
            srdca_local_kernel_size=srdca_local_kernel_size,
            srdca_local_dilation=srdca_local_dilation,
            srdca_norm_kv=srdca_norm_kv,
        )
        # Only build the last attention stage when enabled.
        if self.use_c1_attn:
            self.attn_c1 = MixDecodeBlock(
                dim1=c1_in,
                dim2=tot_channels,
                kv_dims=kv_dims,
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                attn_type=attn_type,
                srdca_qk_dim=srdca_qk_dim,
                srdca_scale_routing=srdca_scale_routing,
                srdca_local_value=srdca_local_value,
                srdca_local_kernel_size=srdca_local_kernel_size,
                srdca_local_dilation=srdca_local_dilation,
                srdca_norm_kv=srdca_norm_kv,
            )
        else:
            self.attn_c1 = None

        # CatKey modules (U-MixFormer style progressive kv update)
        assert len(pool_ratio) == 4, "pool_ratio must have 4 elements (s4->s1)"
        pool_ratio = list(pool_ratio)
        if share_cat_key:
            self.cat_key = CatKey(pool_ratio=pool_ratio, dim=kv_dims)
            self.cat_key1 = self.cat_key2 = self.cat_key3 = self.cat_key4 = self.cat_key
        else:
            self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=kv_dims)
            self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=kv_dims)
            self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=kv_dims)
            # cat_key4 is only needed when we use the c1 attention stage.
            self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=kv_dims) if self.use_c1_attn else None

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
        )

    def forward(self, inputs):
        feats = self._transform_inputs(inputs)  # [c1,c2,c3,c4]
        c1, c2, c3, c4 = feats
        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # stage s4 (query=c4, kv=[c4,c3,c2,c1])
        c_key = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)  # (B,Nk,sumC)
        c4_tok = c4.flatten(2).transpose(1, 2)
        _c4 = self.attn_c4(c4_tok, c_key, h4, w4, h4, w4)
        _c4 = _c4.permute(0, 2, 1).reshape(n, -1, h4, w4)

        # stage s3 (query=c3, kv=[_c4,c3,c2,c1])
        c_key = self.cat_key2([_c4, c3, c2, c1]).flatten(2).transpose(1, 2)
        c3_tok = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_tok, c_key, h4, w4, h3, w3)
        _c3 = _c3.permute(0, 2, 1).reshape(n, -1, h3, w3)

        # stage s2 (query=c2, kv=[_c4,_c3,c2,c1])
        c_key = self.cat_key3([_c4, _c3, c2, c1]).flatten(2).transpose(1, 2)
        c2_tok = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_tok, c_key, h4, w4, h2, w2)
        _c2 = _c2.permute(0, 2, 1).reshape(n, -1, h2, w2)

        # stage s1 (query=c1, kv=[_c4,_c3,_c2,c1])
        if self.use_c1_attn:
            assert self.cat_key4 is not None and self.attn_c1 is not None
            c_key = self.cat_key4([_c4, _c3, _c2, c1]).flatten(2).transpose(1, 2)
            c1_tok = c1.flatten(2).transpose(1, 2)
            _c1 = self.attn_c1(c1_tok, c_key, h4, w4, h1, w1)
            _c1 = _c1.permute(0, 2, 1).reshape(n, -1, h1, w1)
        else:
            _c1 = c1

        # Resize to the same spatial size and fuse
        _c4 = resize(_c4, size=(h1, w1), mode="bilinear", align_corners=False)
        _c3 = resize(_c3, size=(h1, w1), mode="bilinear", align_corners=False)
        _c2 = resize(_c2, size=(h1, w1), mode="bilinear", align_corners=False)
        if _c1.shape[-2:] != (h1, w1):
            _c1 = resize(_c1, size=(h1, w1), mode="bilinear", align_corners=False)

        out = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        out = self.cls_seg(out)
        return out
