# This file provides:
#   1) APFormerHead2AgentBaseline: U-MixFormer(APFormerHead2) decoder with *bias-free* AgentAttention-style
#      cross attention replacing the original CrossAttention.
#   2) APFormerHead2AgentAug: an improved decoder with our proposed Agent-Augmented Mix Attention (A2MA).
#
# Notes:
# - Implemented for OpenMMLab mmsegmentation (mmseg>=1.x, mmcv>=2.x, mmengine>=0.x)
# - Stage-wise list arguments follow order: [s4, s3, s2, s1] (deep -> shallow)

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn

from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init, trunc_normal_
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc


def _as_tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (int(x), int(x))


class DWConv(BaseModule):
    """Depth-wise 3x3 conv used in MixFFN-like blocks."""

    def __init__(self, dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, padding, bias=bias, groups=dim)

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
        # x: [B, N, C]
        b, n, c = x.shape
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MixMlp(BaseModule):
    """MLP with depthwise conv (SegFormer/MiT style) used in APFormer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        dwconv_kernel_size: int = 3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.dwconv = DWConv(hidden_features, kernel_size=dwconv_kernel_size, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

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
    """Align multi-scale features to s4 resolution via AvgPool, then concat on channel.

    pool_ratios order: [s4, s3, s2, s1] (deep -> shallow)
    dims order:        [c4, c3, c2, c1]
    """

    def __init__(self, pool_ratios: Sequence[int], dims: Sequence[int]):
        super().__init__()
        assert len(pool_ratios) == len(dims)
        self.pool_ratios = list(pool_ratios)

        self.sr_list = ModuleList()
        self.pool_list = ModuleList()
        for pr, dim in zip(self.pool_ratios, dims):
            if pr > 1:
                self.pool_list.append(nn.AvgPool2d(kernel_size=pr, stride=pr, ceil_mode=True))
                self.sr_list.append(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True))

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
        assert len(feats_s4_to_s1) == len(self.pool_ratios)
        out = []
        conv_pool_idx = 0
        for i, pr in enumerate(self.pool_ratios):
            x = feats_s4_to_s1[i]
            if pr > 1:
                x = self.pool_list[conv_pool_idx](x)
                x = self.sr_list[conv_pool_idx](x)
                conv_pool_idx += 1
            out.append(x)
        return torch.cat(out, dim=1)


class AgentCrossAttentionNoBias(BaseModule):
    """Bias-free AgentAttention adapted to *cross-attention*.

    This is used as the requested baseline:
      - no positional biases (unlike the official AgentAttention)
      - agent tokens generated by adaptive avg pooling from query features

    Args:
        dim_q (int): query embedding dim (stage channel)
        dim_kv (int): key/value embedding dim (concat channel)
        num_heads (int): heads
        agent_pool_size (Tuple[int,int]): (Ha, Wa) for agent token grid, agent_num=Ha*Wa
        qkv_bias (bool): linear bias in q/kv projections
        attn_drop/proj_drop (float)
        use_dwconv (bool): local detail enhancement on query feature (CPE-like)
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        use_learnable_agent: bool = False,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f"dim_q={dim_q} must be divisible by num_heads={num_heads}"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_pool_size = _as_tuple(agent_pool_size)
        self.agent_pool = nn.AdaptiveAvgPool2d(self.agent_pool_size)

        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(
                in_channels=dim_q,
                out_channels=dim_q,
                kernel_size=dwconv_kernel_size,
                padding=dwconv_kernel_size // 2,
                groups=dim_q,
                bias=True,
            )

        self.use_learnable_agent = use_learnable_agent
        if use_learnable_agent:
            agent_num = self.agent_pool_size[0] * self.agent_pool_size[1]
            self.learnable_agent = nn.Parameter(torch.zeros(1, agent_num, dim_q))
            trunc_normal_(self.learnable_agent, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

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

    def _make_agent_tokens(self, q_tokens: torch.Tensor, hq: int, wq: int) -> torch.Tensor:
        # q_tokens: [B, Nq, C]
        b, nq, c = q_tokens.shape
        q_map = q_tokens.transpose(1, 2).reshape(b, c, hq, wq)
        a = self.agent_pool(q_map).reshape(b, c, -1).transpose(1, 2)  # [B, Na, C]
        if self.use_learnable_agent:
            a = a + self.learnable_agent
        return a

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hw_kv: Tuple[int, int],
        hw_q: Tuple[int, int],
    ) -> torch.Tensor:
        # x_q: [B, Nq, Cq]
        # x_kv: [B, Nk, Ckv]
        b, nq, cq = x_q.shape
        hk, wk = hw_kv
        hq, wq = hw_q

        q = self.q(x_q)  # [B, Nq, Cq]
        kv = self.kv(x_kv)  # [B, Nk, 2*Cq]

        # split heads
        q = q.reshape(b, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Nq, d]
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, h, Nk, d]

        # agent tokens from *query*
        a = self._make_agent_tokens(self.q(x_q), hq, wq)  # [B, Na, Cq]
        na = a.shape[1]
        a = a.reshape(b, na, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Na, d]

        # agent aggregation: A queries K,V
        attn_a = self.softmax((a @ k.transpose(-2, -1)) * self.scale)  # [B, h, Na, Nk]
        attn_a = self.attn_drop(attn_a)
        v_a = attn_a @ v  # [B, h, Na, d]

        # broadcast: Q queries A, values are v_a
        attn_q = self.softmax((q @ a.transpose(-2, -1)) * self.scale)  # [B, h, Nq, Na]
        attn_q = self.attn_drop(attn_q)
        out = attn_q @ v_a  # [B, h, Nq, d]
        out = out.transpose(1, 2).reshape(b, nq, cq)

        # local enhancement on query feature (CPE-like)
        if self.use_dwconv:
            x_map = x_q.transpose(1, 2).reshape(b, cq, hq, wq)
            out = out + self.dwconv(x_map).flatten(2).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AugmentedCrossAgentAttention(BaseModule):
    """Proposed Agent-Augmented Mix Attention (A2MA).

    Motivation: cross-attention is accurate but can be noisy; agent attention is efficient
    but low-rank and may lose fine details. Instead of *hard* gating or simply summing
    two independent attentions, we **augment** the key/value set with agent-summary tokens
    and let a single softmax allocate probability mass between {fine tokens} and {agent tokens}.

    Steps:
      1) Build agent tokens A from pooled query.
      2) Compute agent-summary values V_A = softmax(AK^T)V.
      3) Augment KV: K~=[K;A], V~=[V;V_A].
      4) Output: softmax(Q K~^T) V~.

    This keeps fine-grained matching via K, while enabling a global, low-rank shortcut via A.

    Args are similar to AgentCrossAttentionNoBias.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        use_learnable_agent: bool = False,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f"dim_q={dim_q} must be divisible by num_heads={num_heads}"
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_pool_size = _as_tuple(agent_pool_size)
        self.agent_pool = nn.AdaptiveAvgPool2d(self.agent_pool_size)

        self.use_dwconv = use_dwconv
        if use_dwconv:
            self.dwconv = nn.Conv2d(
                in_channels=dim_q,
                out_channels=dim_q,
                kernel_size=dwconv_kernel_size,
                padding=dwconv_kernel_size // 2,
                groups=dim_q,
                bias=True,
            )

        self.use_learnable_agent = use_learnable_agent
        if use_learnable_agent:
            agent_num = self.agent_pool_size[0] * self.agent_pool_size[1]
            self.learnable_agent = nn.Parameter(torch.zeros(1, agent_num, dim_q))
            trunc_normal_(self.learnable_agent, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

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

    def _make_agent_tokens(self, q_tokens: torch.Tensor, hq: int, wq: int) -> torch.Tensor:
        # q_tokens: [B, Nq, C]
        b, nq, c = q_tokens.shape
        q_map = q_tokens.transpose(1, 2).reshape(b, c, hq, wq)
        a = self.agent_pool(q_map).reshape(b, c, -1).transpose(1, 2)  # [B, Na, C]
        if self.use_learnable_agent:
            a = a + self.learnable_agent
        return a

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hw_kv: Tuple[int, int],
        hw_q: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, cq = x_q.shape
        hk, wk = hw_kv
        hq, wq = hw_q

        q = self.q(x_q)  # [B, Nq, C]
        kv = self.kv(x_kv)  # [B, Nk, 2*C]

        # split heads
        qh = q.reshape(b, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Nq, d]
        kvh = kv.reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kvh[0], kvh[1]  # [B, h, Nk, d]

        # agent tokens from query
        a = self._make_agent_tokens(q, hq, wq)  # [B, Na, C]
        na = a.shape[1]
        a = a.reshape(b, na, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, Na, d]

        # agent-summary values: V_A
        attn_a = self.softmax((a @ k.transpose(-2, -1)) * self.scale)  # [B, h, Na, Nk]
        attn_a = self.attn_drop(attn_a)
        v_a = attn_a @ v  # [B, h, Na, d]

        # augment KV
        k_aug = torch.cat([k, a], dim=2)  # [B, h, Nk+Na, d]
        v_aug = torch.cat([v, v_a], dim=2)  # [B, h, Nk+Na, d]

        # single softmax over augmented tokens
        attn = self.softmax((qh @ k_aug.transpose(-2, -1)) * self.scale)  # [B, h, Nq, Nk+Na]
        attn = self.attn_drop(attn)
        out = attn @ v_aug  # [B, h, Nq, d]
        out = out.transpose(1, 2).reshape(b, nq, cq)

        if self.use_dwconv:
            x_map = x_q.transpose(1, 2).reshape(b, cq, hq, wq)
            out = out + self.dwconv(x_map).flatten(2).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class UMixFormerAgentBlock(BaseModule):
    """A single decoder stage block: LN -> (Attn) -> LN -> (MixMLP).

    This is adapted from the official mmseg U-MixFormer head (APFormerHead2),
    but with pluggable attention modules.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        act_layer: type[nn.Module] = nn.GELU,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        attn_type: str = 'agent',
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        use_learnable_agent: bool = False,
    ):
        super().__init__()
        assert attn_type in ['agent', 'a2ma']

        self.norm_q = norm_layer(dim_q)
        self.norm_kv = norm_layer(dim_kv)
        self.norm_mlp = norm_layer(dim_q)

        if attn_type == 'agent':
            self.attn = AgentCrossAttentionNoBias(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                agent_pool_size=agent_pool_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_dwconv=use_dwconv,
                dwconv_kernel_size=dwconv_kernel_size,
                use_learnable_agent=use_learnable_agent,
            )
        else:
            self.attn = AugmentedCrossAgentAttention(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                agent_pool_size=agent_pool_size,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_dwconv=use_dwconv,
                dwconv_kernel_size=dwconv_kernel_size,
                use_learnable_agent=use_learnable_agent,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim_q,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            dwconv_kernel_size=dwconv_kernel_size,
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

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hw_kv: Tuple[int, int],
        hw_q: Tuple[int, int],
    ) -> torch.Tensor:
        # pre-norm

        x_q = x_q + self.drop_path(self.attn(self.norm_q(x_q), self.norm_kv(x_kv), hw_kv=hw_kv, hw_q=hw_q))
        x_q = x_q + self.drop_path(self.mlp(self.norm_mlp(x_q), h=hw_q[0], w=hw_q[1]))
        return x_q


@MODELS.register_module()
class AAMAHead(BaseDecodeHead):
    """Improved: APFormerHead2 with Agent-Augmented Mix Attention (A2MA)."""

    def __init__(
        self,
        # stage-wise (s4->s1)
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        agent_pool_sizes: Sequence[Union[int, Tuple[int, int]]] = ((2, 2), (3, 3), (5, 5), (7, 7)),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_paths: Sequence[float] = (0.1, 0.1, 0.1, 0.1),
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        use_learnable_agent: bool = False,
        share_cat_key: bool = False,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(num_heads) == 4
        assert len(pool_ratio) == 4
        assert len(agent_pool_sizes) == 4
        assert len(drop_paths) == 4

        c1, c2, c3, c4 = self.in_channels
        tot_channels = sum(self.in_channels)

        self.block_s4 = UMixFormerAgentBlock(
            dim_q=c4,
            dim_kv=tot_channels,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_paths[0],
            agent_pool_size=agent_pool_sizes[0],
            attn_type='a2ma',
            use_dwconv=use_dwconv,
            dwconv_kernel_size=dwconv_kernel_size,
            use_learnable_agent=use_learnable_agent,
        )
        self.block_s3 = UMixFormerAgentBlock(
            dim_q=c3,
            dim_kv=tot_channels,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_paths[1],
            agent_pool_size=agent_pool_sizes[1],
            attn_type='a2ma',
            use_dwconv=use_dwconv,
            dwconv_kernel_size=dwconv_kernel_size,
            use_learnable_agent=use_learnable_agent,
        )
        self.block_s2 = UMixFormerAgentBlock(
            dim_q=c2,
            dim_kv=tot_channels,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_paths[2],
            agent_pool_size=agent_pool_sizes[2],
            attn_type='a2ma',
            use_dwconv=use_dwconv,
            dwconv_kernel_size=dwconv_kernel_size,
            use_learnable_agent=use_learnable_agent,
        )
        self.block_s1 = UMixFormerAgentBlock(
            dim_q=c1,
            dim_kv=tot_channels,
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_paths[3],
            agent_pool_size=agent_pool_sizes[3],
            attn_type='a2ma',
            use_dwconv=use_dwconv,
            dwconv_kernel_size=dwconv_kernel_size,
            use_learnable_agent=use_learnable_agent,
        )

        dims_s4_to_s1 = (c4, c3, c2, c1)
        if share_cat_key:
            self.cat_key = CatKey(pool_ratios=pool_ratio, dims=dims_s4_to_s1)
        else:
            self.cat_key1 = CatKey(pool_ratios=pool_ratio, dims=dims_s4_to_s1)
            self.cat_key2 = CatKey(pool_ratios=pool_ratio, dims=dims_s4_to_s1)
            self.cat_key3 = CatKey(pool_ratios=pool_ratio, dims=dims_s4_to_s1)
            self.cat_key4 = CatKey(pool_ratios=pool_ratio, dims=dims_s4_to_s1)
        self.share_cat_key = share_cat_key

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def _cat_key(self, idx: int, feats_s4_to_s1: List[torch.Tensor]) -> torch.Tensor:
        if self.share_cat_key:
            return self.cat_key(feats_s4_to_s1)
        return getattr(self, f'cat_key{idx}')(feats_s4_to_s1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        c1, c2, c3, c4 = x

        b, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        kv = self._cat_key(1, [c4, c3, c2, c1])
        kv_tokens = kv.flatten(2).transpose(1, 2)
        c4_tokens = c4.flatten(2).transpose(1, 2)
        c4_tokens = self.block_s4(c4_tokens, kv_tokens, hw_kv=(h4, w4), hw_q=(h4, w4))
        c4 = c4_tokens.transpose(1, 2).reshape(b, -1, h4, w4)

        kv = self._cat_key(2, [c4, c3, c2, c1])
        kv_tokens = kv.flatten(2).transpose(1, 2)
        c3_tokens = c3.flatten(2).transpose(1, 2)
        c3_tokens = self.block_s3(c3_tokens, kv_tokens, hw_kv=(h4, w4), hw_q=(h3, w3))
        c3 = c3_tokens.transpose(1, 2).reshape(b, -1, h3, w3)

        kv = self._cat_key(3, [c4, c3, c2, c1])
        kv_tokens = kv.flatten(2).transpose(1, 2)
        c2_tokens = c2.flatten(2).transpose(1, 2)
        c2_tokens = self.block_s2(c2_tokens, kv_tokens, hw_kv=(h4, w4), hw_q=(h2, w2))
        c2 = c2_tokens.transpose(1, 2).reshape(b, -1, h2, w2)

        kv = self._cat_key(4, [c4, c3, c2, c1])
        kv_tokens = kv.flatten(2).transpose(1, 2)
        c1_tokens = c1.flatten(2).transpose(1, 2)
        c1_tokens = self.block_s1(c1_tokens, kv_tokens, hw_kv=(h4, w4), hw_q=(h1, w1))
        c1 = c1_tokens.transpose(1, 2).reshape(b, -1, h1, w1)

        c4_up = resize(c4, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        c3_up = resize(c3, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        c2_up = resize(c2, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        c1_up = c1

        out = self.linear_fuse(torch.cat([c4_up, c3_up, c2_up, c1_up], dim=1))
        out = self.cls_seg(out)
        return out
