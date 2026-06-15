# This file provides:
#   1) A baseline U-MixFormer decode head where Mix-Attention is replaced by a NO-BIAS AgentAttention-style
#      cross-attention (AgentCrossAttentionNoBias).
#   2) An improved decode head with Scale-Decoupled + Soft-Routed Agent Attention and optional Reciprocal Updates.
#
# Notes for configs:
#   - All stage-wise lists follow the order [s4, s3, s2, s1] (deep -> shallow).
#   - Do NOT pass decoder_params dict; pass ablation params as explicit kwargs.

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc


def _as_stage_list(x: Union[int, float, str, Sequence], n: int = 4) -> list:
    """Broadcast a scalar / string to a stage-wise list (s4->s1)."""
    if isinstance(x, (list, tuple)):
        assert len(x) == n, f'Expected length {n} list, got {len(x)}'
        return list(x)
    return [x for _ in range(n)]


class DWConv(BaseModule):
    """Depthwise 3x3 conv used inside FFN."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

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
        # x: [B, N, C]
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwc(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModule):
    """FFN with depthwise conv (as in U-MixFormer)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
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
    """Spatially align multi-stage feature maps to the smallest resolution (usually s4), then concat on channels.

    This follows the spirit of U-MixFormer mix-attention implementation:
      - each input feature is AvgPool'ed with a stage-specific ratio, then a 1x1 Conv(+Norm+Act)
      - outputs are concatenated along channel dim
    """

    def __init__(
        self,
        pool_ratio: Sequence[int],
        dims: Sequence[int],
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ):
        super().__init__()
        assert len(pool_ratio) == len(dims)
        self.pool_ratio = list(pool_ratio)
        self.dims = list(dims)
        self.num_ins = len(dims)

        self.sr_list = ModuleList()
        self.pool_list = ModuleList()
        for r, d in zip(self.pool_ratio, self.dims):
            if r > 1:
                self.pool_list.append(nn.AvgPool2d(r, r, ceil_mode=True))
            else:
                self.pool_list.append(nn.Identity())
            self.sr_list.append(
                ConvModule(
                    d,
                    d,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
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

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == self.num_ins, f'Expected {self.num_ins} inputs, got {len(feats)}'
        outs = []
        for x, pool, conv in zip(feats, self.pool_list, self.sr_list):
            outs.append(conv(pool(x)))
        return torch.cat(outs, dim=1)


class _GridAgentTokens(BaseModule):
    """Generate agent tokens by spatial grid pooling on query features (as in AgentAttention)."""

    def __init__(self, agent_num: int):
        super().__init__()
        pool_size = int(math.sqrt(agent_num))
        if pool_size * pool_size != agent_num:
            raise ValueError(f'agent_num must be a perfect square for grid pooling, got {agent_num}')
        self.pool_size = pool_size
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    def forward(self, q_2d: torch.Tensor) -> torch.Tensor:
        # q_2d: [B, C, H, W]
        a = self.pool(q_2d)  # [B, C, pH, pW]
        a = a.flatten(2).transpose(1, 2)  # [B, Na, C]
        return a


class _ClusterAgentTokens(BaseModule):
    """Generate content-aware agent tokens by soft clustering.

    Given token features Q in R^{B×N×C}, we compute:
        P = softmax( (Q W) / tau )   over N (token dimension)  -> R^{B×A×N}
        A = P Q                      -> R^{B×A×C}
    """

    def __init__(self, dim: int, agent_num: int, tau: float = 1.0):
        super().__init__()
        self.agent_num = int(agent_num)
        self.tau = float(tau)
        self.proj = nn.Linear(dim, self.agent_num, bias=True)

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

    def forward(self, q_tokens: torch.Tensor) -> torch.Tensor:
        # q_tokens: [B, N, C]
        logits = self.proj(q_tokens)  # [B, N, A]
        weights = (logits.transpose(1, 2) / self.tau).softmax(dim=-1)  # [B, A, N]
        agents = weights @ q_tokens  # [B, A, C]
        return agents


class AgentCrossAttentionNoBias(BaseModule):
    """No-bias AgentAttention-style CROSS-attention.

    - Q comes from x (dim_q), K/V come from y (dim_kv).
    - Agent tokens are derived from Q (grid pooling or content clustering).
    - No learnable bias terms (no agent bias tables).

    Complexity: O(Nq * A + A * Nk) per head (A = agent_num).
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        agent_num: int = 49,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        agent_token_type: str = 'grid',  # 'grid' | 'cluster'
        agent_token_tau: float = 1.0,
        use_dwconv_local: bool = True,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q {dim_q} must be divisible by num_heads {num_heads}'
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_num = int(agent_num)

        # Linear projections: no bias
        self.q = nn.Linear(dim_q, dim_q, bias=True)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=True)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_dwconv_local = bool(use_dwconv_local)
        if self.use_dwconv_local:
            self.dwc = nn.Conv2d(dim_q, dim_q, 3, 1, 1, groups=dim_q)

        agent_token_type = agent_token_type.lower()
        self.agent_token_type = agent_token_type
        if agent_token_type == 'grid':
            self.agent_gen = _GridAgentTokens(self.agent_num)
        elif agent_token_type == 'cluster':
            self.agent_gen = _ClusterAgentTokens(dim_q, self.agent_num, tau=agent_token_tau)
        else:
            raise ValueError(f'Unsupported agent_token_type: {agent_token_type}')

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

    def _make_agents(self, q_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # returns: [B, A, C]
        if self.agent_token_type == 'grid':
            q_2d = q_tokens.transpose(1, 2).reshape(q_tokens.shape[0], self.dim_q, H, W)
            return self.agent_gen(q_2d)
        # cluster
        return self.agent_gen(q_tokens)

    def forward(self, x: torch.Tensor, y: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, Nq, Cq], y: [B, Nk, Ckv]
        B, Nq, _ = x.shape
        Nk = y.shape[1]

        q_tokens = self.q(x)  # [B, Nq, C]
        agents = self._make_agents(q_tokens, H, W)  # [B, A, C]

        # reshape to heads
        q = q_tokens.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,Nq,d]
        a = agents.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,A,d]

        kv = self.kv(y).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B,h,Nk,d]

        # agent -> kv
        attn_a = (a * self.scale) @ k.transpose(-2, -1)  # [B,h,A,Nk]
        attn_a = attn_a.softmax(dim=-1)
        attn_a = self.attn_drop(attn_a)
        av = attn_a @ v  # [B,h,A,d]

        # q -> agent
        attn_q = (q * self.scale) @ a.transpose(-2, -1)  # [B,h,Nq,A]
        attn_q = attn_q.softmax(dim=-1)
        attn_q = self.attn_drop(attn_q)

        out = attn_q @ av  # [B,h,Nq,d]
        out = out.transpose(1, 2).reshape(B, Nq, self.dim_q)

        if self.use_dwconv_local:
            x_2d = x.transpose(1, 2).reshape(B, self.dim_q, H, W)
            local = self.dwc(x_2d).flatten(2).transpose(1, 2)
            out = out + local

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class ScaleRoutedAgentCrossAttentionNoBias(BaseModule):
    """Scale-Decoupled + Soft-Routed Agent Cross-Attention (no bias).

    The key/value tensor y is assumed to be a channel-wise concatenation of S sources:
        y = concat(y_1, ..., y_S) along channel dim, where y_s has kv_dims[s] channels.
    We project each y_s independently to K_s/V_s, aggregate values via agents, then
    fuse per-scale agent-values with a softmax routing weight conditioned on agents.

    This avoids scale interference from a single projection on the concatenated feature.
    """

    def __init__(
        self,
        dim_q: int,
        kv_dims: Sequence[int],  # channel split sizes, order must match concat order
        num_heads: int,
        agent_num: int = 49,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        agent_token_type: str = 'grid',
        agent_token_tau: float = 1.0,
        route_temp: float = 1.0,
        use_dwconv_local: bool = True,
        use_scale_routing: bool = True,
    ):
        super().__init__()
        assert dim_q % num_heads == 0
        self.dim_q = int(dim_q)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_q // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.kv_dims = [int(d) for d in kv_dims]
        self.num_scales = len(self.kv_dims)
        self.agent_num = int(agent_num)

        self.q = nn.Linear(self.dim_q, self.dim_q, bias=True)
        self.kv_projs = ModuleList([nn.Linear(d, self.dim_q * 2, bias=True) for d in self.kv_dims])
        self.kv_norms = ModuleList([nn.LayerNorm(d) for d in self.kv_dims])

        self.proj = nn.Linear(self.dim_q, self.dim_q, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_dwconv_local = bool(use_dwconv_local)
        if self.use_dwconv_local:
            self.dwc = nn.Conv2d(self.dim_q, self.dim_q, 3, 1, 1, groups=self.dim_q)

        agent_token_type = agent_token_type.lower()
        self.agent_token_type = agent_token_type
        if agent_token_type == 'grid':
            self.agent_gen = _GridAgentTokens(self.agent_num)
        elif agent_token_type == 'cluster':
            self.agent_gen = _ClusterAgentTokens(self.dim_q, self.agent_num, tau=agent_token_tau)
        else:
            raise ValueError(f'Unsupported agent_token_type: {agent_token_type}')

        self.use_scale_routing = bool(use_scale_routing)
        self.route_temp = float(route_temp)
        if self.use_scale_routing:
            # route vectors: [h, S, d]
            self.route_vec = nn.Parameter(torch.zeros(self.num_heads, self.num_scales, self.head_dim))
            nn.init.trunc_normal_(self.route_vec, std=0.02)

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

    def _make_agents(self, q_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if self.agent_token_type == 'grid':
            q_2d = q_tokens.transpose(1, 2).reshape(q_tokens.shape[0], self.dim_q, H, W)
            return self.agent_gen(q_2d)
        return self.agent_gen(q_tokens)

    def forward(self, x: torch.Tensor, y: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, Nq, Cq=dim_q], y: [B, Nk, sum(kv_dims)]
        B, Nq, _ = x.shape
        Nk = y.shape[1]
        assert y.shape[2] == sum(self.kv_dims), \
            f'Expected y channel {sum(self.kv_dims)}, got {y.shape[2]}'

        q_tokens = self.q(x)  # [B, Nq, C]
        agents = self._make_agents(q_tokens, H, W)  # [B, A, C]

        q = q_tokens.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,Nq,d]
        a = agents.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,A,d]

        # split y into per-scale segments
        ys = torch.split(y, self.kv_dims, dim=2)

        av_list = []
        for y_s, ln_s, proj_s in zip(ys, self.kv_norms, self.kv_projs):
            y_s = ln_s(y_s)
            kv = proj_s(y_s).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # [B,h,Nk,d]

            attn_a = (a * self.scale) @ k.transpose(-2, -1)  # [B,h,A,Nk]
            attn_a = attn_a.softmax(dim=-1)
            attn_a = self.attn_drop(attn_a)
            av = attn_a @ v  # [B,h,A,d]
            av_list.append(av)

        # route fusion across scales
        if self.use_scale_routing:
            # logits: [B,h,A,S]
            logits = torch.einsum('bhad,hsd->bhas', a, self.route_vec) / self.route_temp
            w = logits.softmax(dim=-1)  # [B,h,A,S]
            av_stack = torch.stack(av_list, dim=-2)  # [B,h,A,S,d]
            av_fused = (w.unsqueeze(-1) * av_stack).sum(dim=-2)  # [B,h,A,d]
        else:
            av_fused = sum(av_list)

        # q -> agent
        attn_q = (q * self.scale) @ a.transpose(-2, -1)  # [B,h,Nq,A]
        attn_q = attn_q.softmax(dim=-1)
        attn_q = self.attn_drop(attn_q)

        out = attn_q @ av_fused  # [B,h,Nq,d]
        out = out.transpose(1, 2).reshape(B, Nq, self.dim_q)

        if self.use_dwconv_local:
            x_2d = x.transpose(1, 2).reshape(B, self.dim_q, H, W)
            local = self.dwc(x_2d).flatten(2).transpose(1, 2)
            out = out + local

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AgentMixBlock(BaseModule):
    """A transformer-like block: LN -> AgentCrossAttn -> residual -> LN -> FFN -> residual."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        agent_num: int = 49,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        agent_token_type: str = 'grid',
        agent_token_tau: float = 1.0,
        use_dwconv_local: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.attn = AgentCrossAttentionNoBias(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            agent_num=agent_num,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            agent_token_type=agent_token_type,
            agent_token_tau=agent_token_tau,
            use_dwconv_local=use_dwconv_local,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=dim_q, hidden_features=mlp_hidden_dim, drop=drop)

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

    def forward(self, x: torch.Tensor, kv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm_q(x), self.norm_kv(kv), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ScaleRoutedAgentMixBlock(BaseModule):
    """Block using ScaleRoutedAgentCrossAttentionNoBias (scale-decoupled KV + soft routing)."""

    def __init__(
        self,
        dim_q: int,
        kv_dims: Sequence[int],
        num_heads: int,
        agent_num: int = 49,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        agent_token_type: str = 'grid',
        agent_token_tau: float = 1.0,
        route_temp: float = 1.0,
        use_dwconv_local: bool = True,
        use_scale_routing: bool = True,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.attn = ScaleRoutedAgentCrossAttentionNoBias(
            dim_q=dim_q,
            kv_dims=kv_dims,
            num_heads=num_heads,
            agent_num=agent_num,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            agent_token_type=agent_token_type,
            agent_token_tau=agent_token_tau,
            route_temp=route_temp,
            use_dwconv_local=use_dwconv_local,
            use_scale_routing=use_scale_routing,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=dim_q, hidden_features=mlp_hidden_dim, drop=drop)

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

    def forward(self, x: torch.Tensor, kv_cat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm_q(x), kv_cat, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ReciprocalAgentUpdate(BaseModule):
    """A lightweight reciprocal (reverse) update: update x_q using y_kv.

    Used to mimic 'mutual' interaction idea (MACA-style) in a cheap way.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        agent_num: int = 49,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        agent_token_type: str = 'grid',
        agent_token_tau: float = 1.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.attn = AgentCrossAttentionNoBias(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            agent_num=agent_num,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            agent_token_type=agent_token_type,
            agent_token_tau=agent_token_tau,
            use_dwconv_local=False,  # reverse update already low-res; keep it minimal
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

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

    def forward(self, x_q: torch.Tensor, y_kv: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x_q: [B, Nq, dim_q]
        return x_q + self.drop_path(self.attn(self.norm_q(x_q), self.norm_kv(y_kv), H, W))


@MODELS.register_module()
class UAgentFormer(BaseDecodeHead):
    """Baseline: U-MixFormer with NO-BIAS AgentAttention replacing Mix-Attention."""

    def __init__(
        self,
        # decoder ablations (explicit, no nested dict)
        num_heads: Sequence[int] = (8, 5, 2, 1),  # s4->s1
        mlp_ratio: float = 4.0,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),  # for [c4,c3,c2,c1] alignment
        agent_nums: Sequence[int] = (49, 49, 49, 49),  # s4->s1
        agent_token_types: Union[str, Sequence[str]] = 'grid',  # str or list(s4->s1)
        agent_token_tau: float = 1.0,
        use_dwconv_local: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        cat_norm_cfg: Optional[dict] = None,
        cat_act_cfg: Optional[dict] = None,
        **kwargs,
    ):
        if 'decoder_params' in kwargs:
            raise ValueError('Do not use decoder_params dict; pass decoder args explicitly.')
        super().__init__(input_transform='multiple_select', **kwargs)

        # channels are [c1,c2,c3,c4] in mmseg
        c1, c2, c3, c4 = self.in_channels
        self.c_dims = (c1, c2, c3, c4)
        self.kv_dims = [c4, c3, c2, c1]  # concat order in CatKey
        self.total_channels = sum(self.kv_dims)

        num_heads = list(num_heads)
        agent_nums = list(agent_nums)
        agent_token_types = _as_stage_list(agent_token_types, 4)

        # CatKey (same as official APFormerHead2)
        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)

        # DropPath schedule (uniform or linear). Keep simple: same for all blocks.
        dpr = float(drop_path_rate)

        self.blk4 = AgentMixBlock(
            dim_q=c4,
            dim_kv=self.total_channels,
            num_heads=num_heads[0],
            agent_num=agent_nums[0],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[0],
            agent_token_tau=agent_token_tau,
            use_dwconv_local=use_dwconv_local,
        )
        self.blk3 = AgentMixBlock(
            dim_q=c3,
            dim_kv=self.total_channels,
            num_heads=num_heads[1],
            agent_num=agent_nums[1],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[1],
            agent_token_tau=agent_token_tau,
            use_dwconv_local=use_dwconv_local,
        )
        self.blk2 = AgentMixBlock(
            dim_q=c2,
            dim_kv=self.total_channels,
            num_heads=num_heads[2],
            agent_num=agent_nums[2],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[2],
            agent_token_tau=agent_token_tau,
            use_dwconv_local=use_dwconv_local,
        )
        self.blk1 = AgentMixBlock(
            dim_q=c1,
            dim_kv=self.total_channels,
            num_heads=num_heads[3],
            agent_num=agent_nums[3],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[3],
            agent_token_tau=agent_token_tau,
            use_dwconv_local=use_dwconv_local,
        )

        # Fuse & predict (same as official APFormerHead2)
        self.linear_fuse = ConvModule(
            self.total_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        # x: [c1,c2,c3,c4]
        c1, c2, c3, c4 = x
        B, _, H1, W1 = c1.shape
        _, _, H2, W2 = c2.shape
        _, _, H3, W3 = c3.shape
        _, _, H4, W4 = c4.shape

        # s4
        kv4 = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)  # [B, N4, Csum]
        q4 = c4.flatten(2).transpose(1, 2)
        d4 = self.blk4(q4, kv4, H4, W4)
        d4 = d4.transpose(1, 2).reshape(B, self.kv_dims[0], H4, W4)

        # s3
        kv3 = self.cat_key2([d4, c3, c2, c1]).flatten(2).transpose(1, 2)
        q3 = c3.flatten(2).transpose(1, 2)
        d3 = self.blk3(q3, kv3, H3, W3)
        d3 = d3.transpose(1, 2).reshape(B, self.kv_dims[1], H3, W3)

        # s2
        kv2 = self.cat_key3([d4, d3, c2, c1]).flatten(2).transpose(1, 2)
        q2 = c2.flatten(2).transpose(1, 2)
        d2 = self.blk2(q2, kv2, H2, W2)
        d2 = d2.transpose(1, 2).reshape(B, self.kv_dims[2], H2, W2)

        # s1
        kv1 = self.cat_key4([d4, d3, d2, c1]).flatten(2).transpose(1, 2)
        q1 = c1.flatten(2).transpose(1, 2)
        d1 = self.blk1(q1, kv1, H1, W1)
        d1 = d1.transpose(1, 2).reshape(B, self.kv_dims[3], H1, W1)

        # upsample to s1 for fusion
        d4_up = resize(d4, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)
        d3_up = resize(d3, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)
        d2_up = resize(d2, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)

        fuse = torch.cat([d4_up, d3_up, d2_up, d1], dim=1)
        fuse = self.linear_fuse(fuse)
        out = self.cls_seg(fuse)
        return out


@MODELS.register_module()
class CARAHead(BaseDecodeHead):
    """Improved: Content-aware + Scale-decoupled + Soft-routed Agent Attention, with optional reciprocal updates.

    CARA = Content-aware routed agent attention.
    """

    def __init__(
        self,
        # decoder ablations (explicit)
        num_heads: Sequence[int] = (8, 5, 2, 1),  # s4->s1
        mlp_ratio: float = 4.0,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        agent_nums: Sequence[int] = (64, 64, 64, 64),
        agent_token_types: Union[str, Sequence[str]] = 'cluster',  # default: content-aware
        agent_token_tau: float = 1.0,
        route_temp: float = 1.0,
        use_dwconv_local: bool = True,
        use_scale_routing: bool = True,
        # reciprocal update
        use_reciprocal: bool = True,
        reciprocal_drop_path: float = 0.0,
        # drops
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        cat_norm_cfg: Optional[dict] = None,
        cat_act_cfg: Optional[dict] = None,
        **kwargs,
    ):
        if 'decoder_params' in kwargs:
            raise ValueError('Do not use decoder_params dict; pass decoder args explicitly.')
        super().__init__(input_transform='multiple_select', **kwargs)

        c1, c2, c3, c4 = self.in_channels
        self.c_dims = (c1, c2, c3, c4)
        self.kv_dims = [c4, c3, c2, c1]
        self.total_channels = sum(self.kv_dims)

        num_heads = list(num_heads)
        agent_nums = list(agent_nums)
        agent_token_types = _as_stage_list(agent_token_types, 4)

        # CatKey is kept identical to APFormerHead2
        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dims=self.kv_dims, norm_cfg=cat_norm_cfg, act_cfg=cat_act_cfg)

        dpr = float(drop_path_rate)

        # Scale-decoupled routed blocks
        self.blk4 = ScaleRoutedAgentMixBlock(
            dim_q=c4,
            kv_dims=self.kv_dims,
            num_heads=num_heads[0],
            agent_num=agent_nums[0],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[0],
            agent_token_tau=agent_token_tau,
            route_temp=route_temp,
            use_dwconv_local=use_dwconv_local,
            use_scale_routing=use_scale_routing,
        )
        self.blk3 = ScaleRoutedAgentMixBlock(
            dim_q=c3,
            kv_dims=self.kv_dims,
            num_heads=num_heads[1],
            agent_num=agent_nums[1],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[1],
            agent_token_tau=agent_token_tau,
            route_temp=route_temp,
            use_dwconv_local=use_dwconv_local,
            use_scale_routing=use_scale_routing,
        )
        self.blk2 = ScaleRoutedAgentMixBlock(
            dim_q=c2,
            kv_dims=self.kv_dims,
            num_heads=num_heads[2],
            agent_num=agent_nums[2],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[2],
            agent_token_tau=agent_token_tau,
            route_temp=route_temp,
            use_dwconv_local=use_dwconv_local,
            use_scale_routing=use_scale_routing,
        )
        self.blk1 = ScaleRoutedAgentMixBlock(
            dim_q=c1,
            kv_dims=self.kv_dims,
            num_heads=num_heads[3],
            agent_num=agent_nums[3],
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr,
            agent_token_type=agent_token_types[3],
            agent_token_tau=agent_token_tau,
            route_temp=route_temp,
            use_dwconv_local=use_dwconv_local,
            use_scale_routing=use_scale_routing,
        )

        # Optional reciprocal updates: d4<-d3, d3<-d2, d2<-d1
        self.use_reciprocal = bool(use_reciprocal)
        if self.use_reciprocal:
            self.rev4 = ReciprocalAgentUpdate(
                dim_q=c4, dim_kv=c3,
                num_heads=num_heads[0],
                agent_num=agent_nums[0],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=reciprocal_drop_path,
                agent_token_type=agent_token_types[0],
                agent_token_tau=agent_token_tau,
            )
            self.rev3 = ReciprocalAgentUpdate(
                dim_q=c3, dim_kv=c2,
                num_heads=num_heads[1],
                agent_num=agent_nums[1],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=reciprocal_drop_path,
                agent_token_type=agent_token_types[1],
                agent_token_tau=agent_token_tau,
            )
            self.rev2 = ReciprocalAgentUpdate(
                dim_q=c2, dim_kv=c1,
                num_heads=num_heads[2],
                agent_num=agent_nums[2],
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=reciprocal_drop_path,
                agent_token_type=agent_token_types[2],
                agent_token_tau=agent_token_tau,
            )

        # Fuse & predict
        self.linear_fuse = ConvModule(
            self.total_channels,
            self.channels,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x
        B, _, H1, W1 = c1.shape
        _, _, H2, W2 = c2.shape
        _, _, H3, W3 = c3.shape
        _, _, H4, W4 = c4.shape

        # s4
        kv4 = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2)  # [B, N4, Csum]
        q4 = c4.flatten(2).transpose(1, 2)
        d4_tok = self.blk4(q4, kv4, H4, W4)
        d4 = d4_tok.transpose(1, 2).reshape(B, self.kv_dims[0], H4, W4)

        # s3
        kv3 = self.cat_key2([d4, c3, c2, c1]).flatten(2).transpose(1, 2)
        q3 = c3.flatten(2).transpose(1, 2)
        d3_tok = self.blk3(q3, kv3, H3, W3)
        d3 = d3_tok.transpose(1, 2).reshape(B, self.kv_dims[1], H3, W3)

        # reciprocal: d4 <- pooled(d3)
        if self.use_reciprocal:
            d3_to_4 = F.adaptive_avg_pool2d(d3, (H4, W4)).flatten(2).transpose(1, 2)  # [B,N4,c3]
            d4_tok = d4.flatten(2).transpose(1, 2)
            d4_tok = self.rev4(d4_tok, d3_to_4, H4, W4)
            d4 = d4_tok.transpose(1, 2).reshape(B, self.kv_dims[0], H4, W4)

        # s2
        kv2 = self.cat_key3([d4, d3, c2, c1]).flatten(2).transpose(1, 2)
        q2 = c2.flatten(2).transpose(1, 2)
        d2_tok = self.blk2(q2, kv2, H2, W2)
        d2 = d2_tok.transpose(1, 2).reshape(B, self.kv_dims[2], H2, W2)

        # reciprocal: d3 <- pooled(d2)
        if self.use_reciprocal:
            d2_to_3 = F.adaptive_avg_pool2d(d2, (H3, W3)).flatten(2).transpose(1, 2)  # [B,N3,c2]
            d3_tok = d3.flatten(2).transpose(1, 2)
            d3_tok = self.rev3(d3_tok, d2_to_3, H3, W3)
            d3 = d3_tok.transpose(1, 2).reshape(B, self.kv_dims[1], H3, W3)

        # s1
        kv1 = self.cat_key4([d4, d3, d2, c1]).flatten(2).transpose(1, 2)
        q1 = c1.flatten(2).transpose(1, 2)
        d1_tok = self.blk1(q1, kv1, H1, W1)
        d1 = d1_tok.transpose(1, 2).reshape(B, self.kv_dims[3], H1, W1)

        # reciprocal: d2 <- pooled(d1)  (helps final fusion)
        if self.use_reciprocal:
            d1_to_2 = F.adaptive_avg_pool2d(d1, (H2, W2)).flatten(2).transpose(1, 2)  # [B,N2,c1]
            d2_tok = d2.flatten(2).transpose(1, 2)
            d2_tok = self.rev2(d2_tok, d1_to_2, H2, W2)
            d2 = d2_tok.transpose(1, 2).reshape(B, self.kv_dims[2], H2, W2)

        # upsample for fusion
        d4_up = resize(d4, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)
        d3_up = resize(d3, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)
        d2_up = resize(d2, size=(H1, W1), mode='bilinear', align_corners=self.align_corners)

        fuse = torch.cat([d4_up, d3_up, d2_up, d1], dim=1)
        fuse = self.linear_fuse(fuse)
        out = self.cls_seg(fuse)
        return out
