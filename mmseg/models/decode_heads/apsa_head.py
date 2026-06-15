# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------------------------
# U-MixFormer decoder head with ablation-ready attention variants.
# This file provides a drop-in DecodeHead for MMSegmentation.
# ---------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_init, constant_init, normal_init

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc
from mmseg.registry import MODELS


def _to_4tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int, int, int]:
    if isinstance(x, int):
        return (x, x, x, x)
    assert len(x) == 4, f'Expect length=4, got {len(x)}'
    return (int(x[0]), int(x[1]), int(x[2]), int(x[3]))


def _to_4list(
    x: Union[int, float, Sequence[Union[int, float]]],
    dtype=float,
) -> List[Union[int, float]]:
    if isinstance(x, (int, float)):
        return [dtype(x)] * 4
    assert len(x) == 4, f'Expect length=4, got {len(x)}'
    return [dtype(v) for v in x]


class DWConv(BaseModule):
    """Depthwise 3x3 conv for token mixing (NLC <-> NCHW)."""

    def __init__(self, dim: int, kernel_size: int = 3, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        pad = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, pad, groups=dim)

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

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = nlc_to_nchw(x, (h, w))
        x = self.dwconv(x)
        x = nchw_to_nlc(x)
        return x


class Mlp(BaseModule):
    """MLP with an interleaved depthwise conv (as in U-MixFormer official code)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
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

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModule):
    """Spatially align multi-stage features to the deepest resolution and concat in channel dim.

    Args:
        pool_ratio (Sequence[int]): per-stage pooling ratio in order [s4, s3, s2, s1]
            where s4 is deepest (smallest spatial).
        dims (Sequence[int]): per-stage channels in order [s4, s3, s2, s1].
    """

    def __init__(
        self,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        dims: Sequence[int] = (256, 160, 64, 32),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(pool_ratio) == 4 and len(dims) == 4
        self.pool_ratio = list(pool_ratio)

        # Only create modules for ratios > 1 to satisfy "instantiate only if needed".
        self.sr_list = ModuleList()
        self.pool_list = ModuleList()
        for i, r in enumerate(self.pool_ratio):
            if r > 1:
                self.sr_list.append(nn.Conv2d(dims[i], dims[i], kernel_size=1, stride=1))
                self.pool_list.append(nn.AvgPool2d(r, r, ceil_mode=True))

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

    def forward(self, feats_s4_to_s1: List[torch.Tensor]) -> torch.Tensor:
        """feats list order: [s4, s3, s2, s1] each is (B, C_i, H_i, W_i)."""
        assert len(feats_s4_to_s1) == 4
        outs = []
        cnt = 0
        for i, r in enumerate(self.pool_ratio):
            x = feats_s4_to_s1[i]
            if r > 1:
                x = self.pool_list[cnt](x)
                x = self.sr_list[cnt](x)
                cnt += 1
            outs.append(x)
        # Channel concat, spatial dims are aligned to s4.
        return torch.cat(outs, dim=1)


class LocalPerception(BaseModule):
    """A lightweight local enhancement module (no sigmoid gating).

    This plays a similar *role* to SCASeg's LPM (local perception) but avoids
    explicit gate-based modulation.
    """

    def __init__(self, dim: int, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
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

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x2d = nlc_to_nchw(x, (h, w))
        y = self.dwconv(x2d)
        y = self.act(y)
        y = self.pwconv(y)
        y = nchw_to_nlc(y)
        return y


class _BaseCrossAttention(BaseModule):
    """Base class for attention used inside U-MixFormer-like decoder blocks."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert dim_q % num_heads == 0, f'dim_q {dim_q} must be divisible by num_heads {num_heads}'
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
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

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        raise NotImplementedError


class VanillaCrossAttention(_BaseCrossAttention):
    """Vanilla multi-head cross-attention (baseline for ablation)."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        init_cfg=None,
    ):
        super().__init__(dim_q, dim_kv, num_heads, attn_drop, proj_drop, init_cfg=init_cfg)
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape

        q = self.q(x_q).reshape(b, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(b, nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, nq, self.dim_q)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class StripCrossAttention(_BaseCrossAttention):
    """Strip cross-attention (SCASeg-like): compute attention logits with low-dim Q/K.

    Note: This reduces the cost of QK^T but keeps AV as-is.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        strip_dim: int = 1,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        init_cfg=None,
    ):
        super().__init__(dim_q, dim_kv, num_heads, attn_drop, proj_drop, init_cfg=init_cfg)
        assert strip_dim >= 1
        self.strip_dim = strip_dim
        self.scale = strip_dim ** -0.5

        self.qs = nn.Linear(dim_q, num_heads * strip_dim, bias=qkv_bias)
        self.ks = nn.Linear(dim_kv, num_heads * strip_dim, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape

        q = self.qs(x_q).reshape(b, nq, self.num_heads, self.strip_dim).permute(0, 2, 1, 3)
        k = self.ks(x_kv).reshape(b, nk, self.num_heads, self.strip_dim).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, nq, self.dim_q)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AgentCrossAttention(_BaseCrossAttention):
    """Agent (low-rank) cross-attention (MacFormer-inspired), one direction: Q <- KV.

    O = softmax(Q A^T) softmax(A K^T) V, where A are pooled agent tokens.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        agent_pool_size: int = 2,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        init_cfg=None,
    ):
        super().__init__(dim_q, dim_kv, num_heads, attn_drop, proj_drop, init_cfg=init_cfg)
        assert agent_pool_size >= 1
        self.agent_pool_size = agent_pool_size
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.k = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)
        self.a = nn.Linear(dim_kv, dim_q, bias=qkv_bias)  # project pooled agents to dim_q

        self.pool = nn.AdaptiveAvgPool2d((agent_pool_size, agent_pool_size))

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape
        hk, wk = kv_hw

        q = self.q(x_q).reshape(b, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x_kv).reshape(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x_kv).reshape(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # pooled agents from KV tokens (keep spatial layout)
        kv_2d = x_kv.transpose(1, 2).reshape(b, self.dim_kv, hk, wk)
        agents = self.pool(kv_2d).flatten(2).transpose(1, 2)  # (B, Na, C_kv)
        na = agents.shape[1]
        a = self.a(agents).reshape(b, na, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_ak = (a @ k.transpose(-2, -1)) * self.scale  # (B,h,Na,Nk)
        attn_ak = attn_ak.softmax(dim=-1)
        attn_ak = self.attn_drop(attn_ak)
        agent_v = attn_ak @ v  # (B,h,Na,d)

        attn_qa = (q @ a.transpose(-2, -1)) * self.scale  # (B,h,Nq,Na)
        attn_qa = attn_qa.softmax(dim=-1)
        attn_qa = self.attn_drop(attn_qa)

        out = (attn_qa @ agent_v).transpose(1, 2).reshape(b, nq, self.dim_q)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AgentProjectedStripCrossAttention(_BaseCrossAttention):
    """Our proposal: Agent-Projected Strip Cross-Attention (APSA).

    Key idea:
      logits = (Q_s K_s^T) + lambda * (Q_g K_g^T),
    where Q_s/K_s are low-dim strip projections (r=strip_dim),
    and Q_g/K_g are agent-projected embeddings computed via pooled agent tokens.

    This keeps a full attention map (for accuracy) but injects a global-context
    similarity term in the *same* logits (no hard/gated fusion).
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        strip_dim: int = 4,
        agent_pool_size: int = 2,
        agent_scale: float = 1.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_value_local: bool = False,
        init_cfg=None,
    ):
        super().__init__(dim_q, dim_kv, num_heads, attn_drop, proj_drop, init_cfg=init_cfg)
        assert strip_dim >= 1
        assert agent_pool_size >= 1
        self.strip_dim = strip_dim
        self.agent_pool_size = agent_pool_size
        self.agent_scale = float(agent_scale)
        self.scale = strip_dim ** -0.5

        # strip projections for logits
        self.qs = nn.Linear(dim_q, num_heads * strip_dim, bias=qkv_bias)
        self.ks = nn.Linear(dim_kv, num_heads * strip_dim, bias=qkv_bias)

        # pooled agents projected into strip space
        self.pool = nn.AdaptiveAvgPool2d((agent_pool_size, agent_pool_size))
        self.as_proj = nn.Linear(dim_kv, num_heads * strip_dim, bias=qkv_bias)

        # value projection (full dim_q)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)

        self.use_value_local = bool(use_value_local)
        if self.use_value_local:
            # depthwise conv on projected values (dim_q) in KV spatial domain
            self.v_dwconv = nn.Conv2d(dim_q, dim_q, kernel_size=3, stride=1, padding=1, groups=dim_q)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape
        hk, wk = kv_hw

        # Strip projections
        q_s = self.qs(x_q).reshape(b, nq, self.num_heads, self.strip_dim).permute(0, 2, 1, 3)  # (B,h,Nq,r)
        k_s = self.ks(x_kv).reshape(b, nk, self.num_heads, self.strip_dim).permute(0, 2, 1, 3)  # (B,h,Nk,r)

        # Values (full dim_q)
        v = self.v(x_kv)  # (B,Nk,Cq)
        if self.use_value_local:
            v2d = v.transpose(1, 2).reshape(b, self.dim_q, hk, wk)
            v2d = v2d + self.v_dwconv(v2d)
            v = v2d.flatten(2).transpose(1, 2)
        v = v.reshape(b, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Agents from KV tokens
        kv_2d = x_kv.transpose(1, 2).reshape(b, self.dim_kv, hk, wk)
        agents = self.pool(kv_2d).flatten(2).transpose(1, 2)  # (B,Na,C_kv)
        na = agents.shape[1]
        a_s = self.as_proj(agents).reshape(b, na, self.num_heads, self.strip_dim).permute(0, 2, 1, 3)  # (B,h,Na,r)

        # Project queries/keys into agent subspace
        attn_q_a = (q_s @ a_s.transpose(-2, -1)) * self.scale  # (B,h,Nq,Na)
        attn_q_a = attn_q_a.softmax(dim=-1)
        q_g = attn_q_a @ a_s  # (B,h,Nq,r)

        attn_k_a = (k_s @ a_s.transpose(-2, -1)) * self.scale  # (B,h,Nk,Na)
        attn_k_a = attn_k_a.softmax(dim=-1)
        k_g = attn_k_a @ a_s  # (B,h,Nk,r)

        # Logits: direct strip similarity + agent-projected global similarity
        logits = (q_s @ k_s.transpose(-2, -1)) * self.scale
        if self.agent_scale != 0:
            logits = logits + (q_g @ k_g.transpose(-2, -1)) * (self.scale * self.agent_scale)

        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, nq, self.dim_q)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixFormerDecoderBlock(BaseModule):
    """A decoder block with (cross-)attention + optional local perception + MLP.

    Order (pre-norm):
        x = x + Attn(LN(x), LN(kv))
        x = x + Local(LN(x))              (optional)
        x = x + MLP(LN(x))
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        attn_mode: str = 'apsa',
        strip_dim: int = 4,
        agent_pool_size: int = 2,
        agent_scale: float = 1.0,
        use_local_perception: bool = False,
        use_value_local: bool = False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        attn_mode = attn_mode.lower()
        self.attn_mode = attn_mode
        if attn_mode == 'vanilla':
            self.attn = VanillaCrossAttention(
                dim_q, dim_kv, num_heads,
                attn_drop=attn_drop, proj_drop=drop,
            )
        elif attn_mode == 'sca':
            self.attn = StripCrossAttention(
                dim_q, dim_kv, num_heads,
                strip_dim=strip_dim,
                attn_drop=attn_drop, proj_drop=drop,
            )
        elif attn_mode == 'agent':
            self.attn = AgentCrossAttention(
                dim_q, dim_kv, num_heads,
                agent_pool_size=agent_pool_size,
                attn_drop=attn_drop, proj_drop=drop,
            )
        elif attn_mode == 'apsa':
            self.attn = AgentProjectedStripCrossAttention(
                dim_q, dim_kv, num_heads,
                strip_dim=strip_dim,
                agent_pool_size=agent_pool_size,
                agent_scale=agent_scale,
                attn_drop=attn_drop, proj_drop=drop,
                use_value_local=use_value_local,
            )
        else:
            raise ValueError(
                f'Unsupported attn_mode={attn_mode}. '
                f'Choose from ["vanilla","sca","agent","apsa"].',
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_local_perception = bool(use_local_perception)
        if self.use_local_perception:
            self.norm_local = nn.LayerNorm(dim_q)
            self.local = LocalPerception(dim_q)
        else:
            self.norm_local = None
            self.local = None

        self.norm_mlp = nn.LayerNorm(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=dim_q, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_hw: Tuple[int, int],
        q_hw: Tuple[int, int],
    ) -> torch.Tensor:
        hq, wq = q_hw

        x = x_q + self.drop_path(self.attn(self.norm_q(x_q), self.norm_kv(x_kv), kv_hw))

        if self.use_local_perception:
            x = x + self.drop_path(self.local(self.norm_local(x), hq, wq))

        x = x + self.drop_path(self.mlp(self.norm_mlp(x), hq, wq))
        return x


@MODELS.register_module()
class APSAHead(BaseDecodeHead):
    """Ablation-ready U-MixFormer decode head with improved attention interaction.

    Notes
    -----
    - List parameters are ordered as [s4, s3, s2, s1] (deep -> shallow), as requested.
    - This head keeps the U-MixFormer key/value mixing scheme (CatKey) and stage-wise
      propagation, and replaces the attention block with ablation switches.

    Key ablation knobs (all are init args, no nested dict):
        attn_mode: 'vanilla' | 'sca' | 'agent' | 'apsa'
        strip_dims: per-stage strip dim r (used by 'sca'/'apsa')
        agent_pool_sizes: per-stage pooled grid size g (agents= g*g)
        agent_scale: lambda for APSA agent-projected similarity (0 disables the term)
        use_local_perception: add a local conv residual sub-block
        use_value_local: add depthwise conv on values inside APSA
        use_decoder_propagation: whether to feed decoded features into next-stage KV mix
    """

    def __init__(
        self,
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        attn_mode: str = 'apsa',
        strip_dims: Union[int, Sequence[int]] = (4, 4, 2, 2),
        agent_pool_sizes: Union[int, Sequence[int]] = (2, 2, 2, 2),
        agent_scale: float = 1.0,
        use_local_perception: Union[bool, Sequence[bool]] = False,
        use_value_local: Union[bool, Sequence[bool]] = False,
        use_decoder_propagation: bool = True,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path_rate: Union[float, Sequence[float]] = 0.1,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        # In MMSeg, in_channels are usually [c1,c2,c3,c4] (shallow->deep).
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = sum(self.in_channels)

        # Normalize per-stage args (order: s4->s1).
        num_heads = _to_4tuple(num_heads)
        pool_ratio = _to_4tuple(pool_ratio)
        strip_dims = _to_4tuple(strip_dims)
        agent_pool_sizes = _to_4tuple(agent_pool_sizes)

        if isinstance(use_local_perception, bool):
            use_local_perception = [use_local_perception] * 4
        else:
            assert len(use_local_perception) == 4
            use_local_perception = list(use_local_perception)

        if isinstance(use_value_local, bool):
            use_value_local = [use_value_local] * 4
        else:
            assert len(use_value_local) == 4
            use_value_local = list(use_value_local)

        drop_path_list = _to_4list(drop_path_rate, dtype=float)

        # Stage order: s4(c4) -> s3(c3) -> s2(c2) -> s1(c1)
        self.block_s4 = MixFormerDecoderBlock(
            dim_q=c4_in, dim_kv=tot_channels, num_heads=num_heads[0],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path_list[0],
            attn_mode=attn_mode, strip_dim=strip_dims[0],
            agent_pool_size=agent_pool_sizes[0], agent_scale=agent_scale,
            use_local_perception=use_local_perception[0],
            use_value_local=use_value_local[0],
        )

        self.block_s3 = MixFormerDecoderBlock(
            dim_q=c3_in, dim_kv=tot_channels, num_heads=num_heads[1],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path_list[1],
            attn_mode=attn_mode, strip_dim=strip_dims[1],
            agent_pool_size=agent_pool_sizes[1], agent_scale=agent_scale,
            use_local_perception=use_local_perception[1],
            use_value_local=use_value_local[1],
        )

        self.block_s2 = MixFormerDecoderBlock(
            dim_q=c2_in, dim_kv=tot_channels, num_heads=num_heads[2],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path_list[2],
            attn_mode=attn_mode, strip_dim=strip_dims[2],
            agent_pool_size=agent_pool_sizes[2], agent_scale=agent_scale,
            use_local_perception=use_local_perception[2],
            use_value_local=use_value_local[2],
        )

        self.block_s1 = MixFormerDecoderBlock(
            dim_q=c1_in, dim_kv=tot_channels, num_heads=num_heads[3],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path_list[3],
            attn_mode=attn_mode, strip_dim=strip_dims[3],
            agent_pool_size=agent_pool_sizes[3], agent_scale=agent_scale,
            use_local_perception=use_local_perception[3],
            use_value_local=use_value_local[3],
        )

        # CatKey modules (always align to deepest resolution).
        self.cat_key_s4 = CatKey(pool_ratio=pool_ratio, dims=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key_s3 = CatKey(pool_ratio=pool_ratio, dims=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key_s2 = CatKey(pool_ratio=pool_ratio, dims=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key_s1 = CatKey(pool_ratio=pool_ratio, dims=(c4_in, c3_in, c2_in, c1_in))

        self.use_decoder_propagation = bool(use_decoder_propagation)

        # Fuse multi-scale decoded features (concat in channel -> embed_dim -> pred)
        # Reuse BaseDecodeHead's predictor (`conv_seg`) via `cls_seg` to avoid
        # instantiating an extra pred conv. This also matches the requirement:
        # "modules not needed should not be instantiated".

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs are multi-level features in order [c1,c2,c3,c4] by MMSeg convention.
        feats = self._transform_inputs(inputs)
        c1, c2, c3, c4 = feats

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        kv_hw = (h4, w4)

        # --- s4 (deepest) ---
        kv = self.cat_key_s4([c4, c3, c2, c1])  # (B, Ctot, h4, w4)
        kv = kv.flatten(2).transpose(1, 2)  # (B, Nk, Ctot)
        q4 = c4.flatten(2).transpose(1, 2)  # (B, Nq4, C4)
        out4 = self.block_s4(q4, kv, kv_hw=kv_hw, q_hw=(h4, w4))
        out4_map = out4.transpose(1, 2).reshape(n, -1, h4, w4)

        # --- s3 ---
        if self.use_decoder_propagation:
            kv_s3_in = [out4_map, c3, c2, c1]
        else:
            kv_s3_in = [c4, c3, c2, c1]
        kv = self.cat_key_s3(kv_s3_in).flatten(2).transpose(1, 2)
        q3 = c3.flatten(2).transpose(1, 2)
        out3 = self.block_s3(q3, kv, kv_hw=kv_hw, q_hw=(h3, w3))
        out3_map = out3.transpose(1, 2).reshape(n, -1, h3, w3)

        # --- s2 ---
        if self.use_decoder_propagation:
            kv_s2_in = [out4_map, out3_map, c2, c1]
        else:
            kv_s2_in = [c4, c3, c2, c1]
        kv = self.cat_key_s2(kv_s2_in).flatten(2).transpose(1, 2)
        q2 = c2.flatten(2).transpose(1, 2)
        out2 = self.block_s2(q2, kv, kv_hw=kv_hw, q_hw=(h2, w2))
        out2_map = out2.transpose(1, 2).reshape(n, -1, h2, w2)

        # --- s1 (shallowest) ---
        if self.use_decoder_propagation:
            kv_s1_in = [out4_map, out3_map, out2_map, c1]
        else:
            kv_s1_in = [c4, c3, c2, c1]
        kv = self.cat_key_s1(kv_s1_in).flatten(2).transpose(1, 2)
        q1 = c1.flatten(2).transpose(1, 2)
        out1 = self.block_s1(q1, kv, kv_hw=kv_hw, q_hw=(h1, w1))
        out1_map = out1.transpose(1, 2).reshape(n, -1, h1, w1)

        # Upsample all decoded features to the finest (c1) resolution and fuse
        out4_up = resize(out4_map, size=(h1, w1), mode='bilinear', align_corners=False)
        out3_up = resize(out3_map, size=(h1, w1), mode='bilinear', align_corners=False)
        out2_up = resize(out2_map, size=(h1, w1), mode='bilinear', align_corners=False)

        fused = torch.cat([out4_up, out3_up, out2_up, out1_map], dim=1)
        fused = self.linear_fuse(fused)
        logits = self.cls_seg(fused)
        return logits
