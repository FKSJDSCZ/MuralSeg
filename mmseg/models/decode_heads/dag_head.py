# Copyright (c) OpenMMLab.
# This file implements a dense, multi-stage mix-attention decoder head
# following the user's proposed DAG-style decoding schedule.
#
# Usage: place this file under mmseg/models/decode_heads/ and add it to
# mmseg/models/decode_heads/__init__.py (or import by config with custom_imports).
#
# The stage-wise list parameters follow the order: stage4 -> stage1 (deep -> shallow).

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import nlc_to_nchw, nchw_to_nlc, resize


def _to_2tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(x, int):
        return (x, x)
    assert len(x) == 2
    return (int(x[0]), int(x[1]))


def _expand_to_4stage(
    value,
    name: str,
    *,
    allow_none: bool = False,
):
    """Expand a scalar/tuple to a 4-stage list (stage4->stage1)."""
    if allow_none and value is None:
        return [None, None, None, None]
    if isinstance(value, (list, tuple)):
        if len(value) == 4:
            return list(value)
    # broadcast
    return [value, value, value, value]


class DWConv(BaseModule):
    """Depth-wise conv for token features in MLP, from U-MixFormer head."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

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
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModule):
    """FFN used in U-MixFormer head: Linear -> DWConv -> GELU -> Linear."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
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
    """Pool+Conv aligner to build a unified Key/Value feature (stage4 resolution)."""

    def __init__(self, pool_ratio: Sequence[int], dim: Sequence[int]):
        """
        Args:
            pool_ratio (list[int]): pooling ratios for [stage4, stage3, stage2, stage1].
            dim (list[int]): channel dims for [stage4, stage3, stage2, stage1].
        """
        super().__init__()
        assert len(pool_ratio) == 4, 'pool_ratio must be length-4: stage4->stage1'
        assert len(dim) == 4, 'dim must be length-4: stage4->stage1'

        self.pool_ratio = list(pool_ratio)
        self.dim = list(dim)

        self.sr_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        for i in range(4):
            self.sr_list.append(nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1))
            # AvgPool2d with ceil_mode to be robust w.r.t. padding/divisibility.
            self.pool_list.append(
                nn.AvgPool2d(kernel_size=self.pool_ratio[i], stride=self.pool_ratio[i], ceil_mode=True),
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

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            xs (list[Tensor]): [x_stage4, x_stage3, x_stage2, x_stage1] in NCHW.

        Returns:
            Tensor: concatenated KV feature in NCHW, spatially aligned to stage4.
        """
        assert len(xs) == 4, 'CatKey expects 4 tensors: stage4->stage1'
        out = []
        for i in range(4):
            x = xs[i]
            x = self.pool_list[i](x)
            x = self.sr_list[i](x)
            out.append(x)
        return torch.cat(out, dim=1)


class CrossAttention(BaseModule):
    """Vanilla cross-attention used in U-MixFormer head (query from x, KV from y)."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q {dim_q} must be divisible by num_heads {num_heads}'
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): query tokens [B, Nq, Cq]
            y (Tensor): kv tokens [B, Nk, Ckv]

        Returns:
            Tensor: output tokens [B, Nq, Cq]
        """
        B, Nq, Cq = x.shape
        _, Nk, _ = y.shape

        q = self.q(x).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Nk, 2, self.num_heads, Cq // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, Nk, hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, Nq, Nk]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AgentCrossAttention(BaseModule):
    """Agent Attention adapted to CROSS-attention (Q from x, KV from y).

    This is a cross-attention version inspired by the official AgentAttention
    implementation (agent_attention.py). It keeps the same two-step aggregation
    (agent->KV) and broadcast (Q->agent) pattern, and supports an ablation to
    enable/disable additive position biases.

    Notes:
        * We assume KV tokens come from a 2D grid (H_kv, W_kv), e.g. stage4
          feature resolution after CatKey pooling.
        * Agent tokens are produced by pooling Q features to downstream_agent_shape.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        agent_num: int = 49,
        downstream_agent_shape: Tuple[int, int] = (7, 7),
        scale: float = -0.5,
        use_additive_bias: bool = True,
        bias_base_hw: Tuple[int, int] = (7, 7),
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q {dim_q} must be divisible by num_heads {num_heads}'
        assert int(agent_num ** 0.5) ** 2 == agent_num, 'agent_num should be a perfect square (e.g., 49)'
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** scale

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_num = agent_num
        self.pool_size = int(agent_num ** 0.5)
        self.downstream_agent_shape = _to_2tuple(downstream_agent_shape)
        self.pool = nn.AdaptiveAvgPool2d(output_size=self.downstream_agent_shape)
        self.softmax = nn.Softmax(dim=-1)

        self.use_additive_bias = bool(use_additive_bias)
        self.bias_base_hw = _to_2tuple(bias_base_hw)

        self.use_dwc = bool(use_dwc)
        if self.use_dwc:
            self.dwc = nn.Conv2d(
                in_channels=dim_q,
                out_channels=dim_q,
                kernel_size=dwc_kernel_size,
                padding=dwc_kernel_size // 2,
                groups=dim_q,
            )

        # Additive biases (optional): following the official AgentAttention design.
        if self.use_additive_bias:
            # an/na: base 2D bias (learned at bias_base_hw, interpolated to target HW)
            bh, bw = self.bias_base_hw
            self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, bh, bw))
            self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, bh, bw))

            # decomposed height/width biases (also interpolated to target HW)
            # (using bias_base_hw as "window_size" base)
            self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, bh, 1))
            self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, bw))
            self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, bh, 1, agent_num))
            self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, bw, agent_num))

            trunc_normal_(self.an_bias, std=0.02)
            trunc_normal_(self.na_bias, std=0.02)
            trunc_normal_(self.ah_bias, std=0.02)
            trunc_normal_(self.aw_bias, std=0.02)
            trunc_normal_(self.ha_bias, std=0.02)
            trunc_normal_(self.wa_bias, std=0.02)

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

    def _agent_to_kv_bias(
        self, b: int, H_kv: int, W_kv: int, downstream_agent_num: int,
    ) -> torch.Tensor:
        """Return bias for agent->KV attention: [B, heads, A, Nk]."""
        # interpolate KV hw (agent_num as channel dim)
        position_bias1 = F.interpolate(self.an_bias, size=(H_kv, W_kv), mode='bilinear')  # [heads, agent_num, Hk, Wk]
        position_bias1 = position_bias1.reshape(
            self.num_heads, self.pool_size, self.pool_size, H_kv * W_kv,
        ).permute(0, 3, 1, 2)  # [heads, Nk, ps, ps]
        # interpolate agent grid to downstream_agent_shape
        position_bias1 = F.interpolate(
            position_bias1, size=self.downstream_agent_shape, mode='bilinear',
        )  # [heads, Nk, Ah, Aw]
        position_bias1 = position_bias1.reshape(
            self.num_heads, H_kv * W_kv, downstream_agent_num,
        ).permute(0, 2, 1)  # [heads, A, Nk]
        position_bias1 = position_bias1.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, heads, A, Nk]

        position_bias2 = F.interpolate(
            (self.ah_bias + self.aw_bias).squeeze(0), size=(H_kv, W_kv), mode='bilinear',
        )  # [heads, agent_num, Hk, Wk]
        position_bias2 = position_bias2.reshape(
            self.num_heads, self.pool_size, self.pool_size, H_kv * W_kv,
        ).permute(0, 3, 1, 2)  # [heads, Nk, ps, ps]
        position_bias2 = F.interpolate(
            position_bias2, size=self.downstream_agent_shape, mode='bilinear',
        )  # [heads, Nk, Ah, Aw]
        position_bias2 = position_bias2.reshape(
            self.num_heads, H_kv * W_kv, downstream_agent_num,
        ).permute(0, 2, 1)  # [heads, A, Nk]
        position_bias2 = position_bias2.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, heads, A, Nk]

        return position_bias1 + position_bias2

    def _q_to_agent_bias(
        self, b: int, H_q: int, W_q: int, downstream_agent_num: int,
    ) -> torch.Tensor:
        """Return bias for Q->agent attention: [B, heads, Nq, A]."""
        agent_bias1 = F.interpolate(
            self.na_bias, size=(H_q, W_q), mode='bilinear',
        )  # [heads, agent_num, Hq, Wq]
        agent_bias1 = agent_bias1.reshape(
            self.num_heads, self.pool_size, self.pool_size, H_q * W_q,
        ).permute(0, 3, 1, 2)  # [heads, Nq, ps, ps]
        agent_bias1 = F.interpolate(
            agent_bias1, size=self.downstream_agent_shape, mode='bilinear',
        )  # [heads, Nq, Ah, Aw]
        agent_bias1 = agent_bias1.reshape(
            self.num_heads, H_q * W_q, downstream_agent_num,
        )  # [heads, Nq, A]
        agent_bias1 = agent_bias1.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, heads, Nq, A]

        agent_bias2 = (self.ha_bias + self.wa_bias).squeeze(0).permute(0, 3, 1, 2)  # [heads, agent_num, bh, bw]
        agent_bias2 = F.interpolate(agent_bias2, size=(H_q, W_q), mode='bilinear')  # [heads, agent_num, Hq, Wq]
        agent_bias2 = agent_bias2.reshape(
            self.num_heads, self.pool_size, self.pool_size, H_q * W_q,
        ).permute(0, 3, 1, 2)  # [heads, Nq, ps, ps]
        agent_bias2 = F.interpolate(
            agent_bias2, size=self.downstream_agent_shape, mode='bilinear',
        )  # [heads, Nq, Ah, Aw]
        agent_bias2 = agent_bias2.reshape(
            self.num_heads, H_q * W_q, downstream_agent_num,
        )  # [heads, Nq, A]
        agent_bias2 = agent_bias2.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, heads, Nq, A]

        return agent_bias1 + agent_bias2

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        H_kv: int,
        W_kv: int,
        H_q: int,
        W_q: int,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): query tokens [B, Nq, Cq]
            y (Tensor): kv tokens [B, Nk, Ckv]
            H_kv, W_kv: spatial size for KV (Nk = H_kv * W_kv)
            H_q, W_q: spatial size for Q (Nq = H_q * W_q)

        Returns:
            Tensor: output tokens [B, Nq, Cq]
        """
        b, Nq, Cq = x.shape
        _, Nk, _ = y.shape
        assert Nq == H_q * W_q, 'Q token length must match H_q*W_q'
        assert Nk == H_kv * W_kv, 'KV token length must match H_kv*W_kv'

        num_heads = self.num_heads
        head_dim = Cq // num_heads

        q = self.q(x)  # [B, Nq, Cq]
        kv = self.kv(y).reshape(b, Nk, 2, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, Nk, hd]

        downstream_agent_num = self.downstream_agent_shape[0] * self.downstream_agent_shape[1]
        # agent tokens pooled from Q (as in official AgentAttention)
        agent_tokens = self.pool(q.reshape(b, H_q, W_q, Cq).permute(0, 3, 1, 2))  # [B, Cq, Ah, Aw]
        agent_tokens = agent_tokens.reshape(b, Cq, -1).permute(0, 2, 1)  # [B, A, Cq]

        q = q.reshape(b, Nq, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, heads, Nq, hd]
        agent_tokens = agent_tokens.reshape(b, downstream_agent_num, num_heads, head_dim).permute(
            0, 2, 1, 3,
        )  # [B, heads, A, hd]

        if self.use_additive_bias:
            position_bias = self._agent_to_kv_bias(b, H_kv, W_kv, downstream_agent_num)  # [B, heads, A, Nk]
        else:
            position_bias = 0

        agent_attn = self.softmax(
            (agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias,
        )  # [B, heads, A, Nk]
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v  # [B, heads, A, hd]

        if self.use_additive_bias:
            agent_bias = self._q_to_agent_bias(b, H_q, W_q, downstream_agent_num)  # [B, heads, Nq, A]
        else:
            agent_bias = 0

        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)  # [B, heads, Nq, A]
        q_attn = self.attn_drop(q_attn)
        out = q_attn @ agent_v  # [B, heads, Nq, hd]

        out = out.transpose(1, 2).reshape(b, Nq, Cq)

        if self.use_dwc:
            # Add a depth-wise conv term from V (KV resolution) and upsample to Q resolution.
            v_map = v.transpose(1, 2).reshape(b, Nk, Cq).transpose(1, 2).reshape(b, Cq, H_kv, W_kv)  # [B, Cq, Hk, Wk]
            v_map = self.dwc(v_map)  # [B, Cq, Hk, Wk]
            v_map = F.interpolate(v_map, size=(H_q, W_q), mode='bilinear', align_corners=False)
            v_token = v_map.flatten(2).transpose(1, 2)  # [B, Nq, Cq]
            out = out + v_token

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MixAttnBlock(BaseModule):
    """A transformer-like block for attn(Q,KV): LN + Attn + FFN (+DropPath)."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        attn_type: str = 'cross',  # 'cross' or 'agent'
        # Agent ablations
        agent_num: int = 49,
        downstream_agent_shape: Tuple[int, int] = (7, 7),
        use_agent_bias: bool = True,
        agent_bias_base_hw: Tuple[int, int] = (7, 7),
        use_agent_dwc: bool = True,
        # common
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.attn_type = attn_type

        self.norm_q = norm_layer(dim_q)
        self.norm_kv = norm_layer(dim_kv)

        if attn_type == 'cross':
            self.attn = CrossAttention(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif attn_type == 'agent':
            self.attn = AgentCrossAttention(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                agent_num=agent_num,
                downstream_agent_shape=downstream_agent_shape,
                use_additive_bias=use_agent_bias,
                bias_base_hw=agent_bias_base_hw,
                use_dwc=use_agent_dwc,
            )
        else:
            raise ValueError(f'Unsupported attn_type={attn_type}. Use "cross" or "agent".')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm_ffn = norm_layer(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=dim_q, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x_q: torch.Tensor,  # [B, Nq, Cq]
        x_kv: torch.Tensor,  # [B, Nk, Ckv]
        H_kv: int,
        W_kv: int,
        H_q: int,
        W_q: int,
    ) -> torch.Tensor:
        if self.attn_type == 'agent':
            x_q = x_q + self.drop_path(
                self.attn(self.norm_q(x_q), self.norm_kv(x_kv), H_kv=H_kv, W_kv=W_kv, H_q=H_q, W_q=W_q),
            )
        else:
            x_q = x_q + self.drop_path(self.attn(self.norm_q(x_q), self.norm_kv(x_kv)))

        x_q = x_q + self.drop_path(self.mlp(self.norm_ffn(x_q), H_q, W_q))
        return x_q


@MODELS.register_module()
class DAGHead(BaseDecodeHead):
    """Dense Mix-Attention Decoder Head.

    Encoder stage outputs: e1, e2, e3, e4 (stage1->stage4; shallow->deep).
    We use stage4-resolution CatKey to build KV, and apply attn(Q,KV) blocks
    following the schedule:

    Stage-1:
        d14 = attn(e4, catkey([e4,e3,e2,e1]))
        d13 = attn(e3, catkey([d14,e3,e2,e1]))
        d12 = attn(e2, catkey([d14,d13,e2,e1]))
        d11 = attn(e1, catkey([d14,d13,d12,e1]))
    Stage-2:
        d23 = attn(e3, catkey([d14,d13,e2,e1]))
        d22 = attn(e2, catkey([d14,d23,d12,e1]))
        d21 = attn(e1, catkey([d14,d23,d22,d11]))
    Stage-3:
        d32 = attn(e2, catkey([d14,d23,d22,e1]))
        d31 = attn(e1, catkey([d14,d23,d32,d21]))
    Stage-4:
        d41 = attn(e1, catkey([d14,d23,d32,d31]))

    Output (ablation):
        - "d41": use d41 (stage1 resolution) as the head feature.
        - "cat": use catkey([d14,d23,d32,d41]) (stage4 resolution), then upsample to stage1.

    Notes:
        * Stage-wise list parameters order: stage4->stage1 (deep->shallow).
        * All ablation parameters are explicit in __init__ (no nested dict).
    """

    def __init__(
        self,
        # CatKey
        pool_ratio: Sequence[int] = (1, 2, 4, 8),  # stage4->stage1
        # Attn/Block settings (stage-wise)
        num_heads: Sequence[int] = (8, 5, 2, 1),  # stage4->stage1
        mlp_ratio: Union[float, Sequence[float]] = 4.0,
        qkv_bias: bool = True,
        # Attention type ablations
        attn_type: str = 'cross',  # 'cross' or 'agent'
        # Agent Attention ablations (used only when attn_type='agent')
        agent_num: int = 49,
        downstream_agent_shape: Union[Tuple[int, int], Sequence[Tuple[int, int]]] = (7, 7),
        use_agent_bias: bool = True,
        agent_bias_base_hw: Tuple[int, int] = (7, 7),
        use_agent_dwc: bool = True,
        # Output ablation
        out_mode: str = 'd41',  # 'd41' or 'cat'
        # drops
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.1,
        fuse_norm_cfg: Optional[dict] = None,
        fuse_act_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        # Normalize stage-wise configs (stage4->stage1)
        num_heads = list(num_heads)
        assert len(num_heads) == 4, 'num_heads must be length-4 (stage4->stage1)'
        mlp_ratio = _expand_to_4stage(mlp_ratio, 'mlp_ratio')
        drop_path_rate = _expand_to_4stage(drop_path_rate, 'drop_path_rate')

        downstream_agent_shape = _expand_to_4stage(downstream_agent_shape, 'downstream_agent_shape')

        # in_channels from BaseDecodeHead are typically [c1,c2,c3,c4] (stage1->stage4)
        c1, c2, c3, c4 = self.in_channels  # stage1->stage4
        dims = [c4, c3, c2, c1]  # stage4->stage1
        self._dims_stage4_to_1 = dims
        self._dims_stage1_to_4 = [c1, c2, c3, c4]

        # CatKey always expects inputs [stage4,stage3,stage2,stage1]
        self.cat_key_d14 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d13 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d12 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d11 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d23 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d22 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d21 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d32 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d31 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_d41 = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.cat_key_fuse = CatKey(pool_ratio=pool_ratio, dim=dims)
        self.dim_kv = sum(dims)

        self.attn_type = attn_type
        self.out_mode = out_mode
        assert self.out_mode in ['d41', 'cat'], f'out_mode must be "d41" or "cat", got {out_mode}'
        assert self.attn_type in ['cross', 'agent'], f'attn_type must be "cross" or "agent", got {attn_type}'

        # Build dense blocks (stage4->stage1): [1,2,3,4] blocks respectively.
        blocks_per_stage = [1, 2, 3, 4]  # stage4->stage1
        self.blocks = ModuleList()
        for stage_idx in range(4):
            dim_q = dims[stage_idx]
            heads = num_heads[stage_idx]
            stage_mlp_ratio = float(mlp_ratio[stage_idx])
            stage_drop_path = float(drop_path_rate[stage_idx])
            stage_agent_shape = _to_2tuple(downstream_agent_shape[stage_idx])

            stage_blocks = ModuleList()
            for _ in range(blocks_per_stage[stage_idx]):
                stage_blocks.append(
                    MixAttnBlock(
                        dim_q=dim_q,
                        dim_kv=self.dim_kv,
                        num_heads=heads,
                        attn_type=self.attn_type,
                        agent_num=agent_num,
                        downstream_agent_shape=stage_agent_shape,
                        use_agent_bias=use_agent_bias,
                        agent_bias_base_hw=agent_bias_base_hw,
                        use_agent_dwc=use_agent_dwc,
                        mlp_ratio=stage_mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=stage_drop_path,
                    ),
                )
            self.blocks.append(stage_blocks)

        # Optional projection (fuse) to self.channels for cls_seg
        # Determine the feature channel of the selected output mode.
        if self.out_mode == 'd41':
            out_in_channels = c1
        else:
            out_in_channels = self.dim_kv

        # Only instantiate fuse module when needed.
        self.fuse = None
        if out_in_channels != self.channels:
            self.fuse = ConvModule(
                in_channels=out_in_channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=fuse_norm_cfg if fuse_norm_cfg is not None else self.norm_cfg,
                act_cfg=fuse_act_cfg if fuse_act_cfg is not None else self.act_cfg,
            )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs are [e1,e2,e3,e4] (stage1->stage4) after transform.
        e1, e2, e3, e4 = self._transform_inputs(inputs)

        B = e1.shape[0]
        h1, w1 = e1.shape[2:]
        h2, w2 = e2.shape[2:]
        h3, w3 = e3.shape[2:]
        h4, w4 = e4.shape[2:]

        # Precompute encoder tokens (Q tokens for all blocks)
        e1_nlc = nchw_to_nlc(e1)
        e2_nlc = nchw_to_nlc(e2)
        e3_nlc = nchw_to_nlc(e3)
        e4_nlc = nchw_to_nlc(e4)

        # ---------- Stage 1 ----------
        kv14_nlc = nchw_to_nlc(self.cat_key_d14([e4, e3, e2, e1]))
        d14 = nlc_to_nchw(self.blocks[0][0](e4_nlc, kv14_nlc, H_kv=h4, W_kv=w4, H_q=h4, W_q=w4), (h4, w4))

        kv13_nlc = nchw_to_nlc(self.cat_key_d13([d14, e3, e2, e1]))
        d13 = nlc_to_nchw(self.blocks[1][0](e3_nlc, kv13_nlc, H_kv=h4, W_kv=w4, H_q=h3, W_q=w3), (h3, w3))

        kv12_nlc = nchw_to_nlc(self.cat_key_d12([d14, d13, e2, e1]))
        d12 = nlc_to_nchw(self.blocks[2][0](e2_nlc, kv12_nlc, H_kv=h4, W_kv=w4, H_q=h2, W_q=w2), (h2, w2))

        kv11_nlc = nchw_to_nlc(self.cat_key_d11([d14, d13, d12, e1]))
        d11 = nlc_to_nchw(self.blocks[3][0](e1_nlc, kv11_nlc, H_kv=h4, W_kv=w4, H_q=h1, W_q=w1), (h1, w1))

        # ---------- Stage 2 ----------
        kv23_nlc = nchw_to_nlc(self.cat_key_d23([d14, d13, e2, e1]))
        d23 = nlc_to_nchw(self.blocks[1][1](e3_nlc, kv23_nlc, H_kv=h4, W_kv=w4, H_q=h3, W_q=w3), (h3, w3))

        kv22_nlc = nchw_to_nlc(self.cat_key_d22([d14, d23, d12, e1]))
        d22 = nlc_to_nchw(self.blocks[2][1](e2_nlc, kv22_nlc, H_kv=h4, W_kv=w4, H_q=h2, W_q=w2), (h2, w2))

        kv21_nlc = nchw_to_nlc(self.cat_key_d21([d14, d23, d22, d11]))
        d21 = nlc_to_nchw(self.blocks[3][1](e1_nlc, kv21_nlc, H_kv=h4, W_kv=w4, H_q=h1, W_q=w1), (h1, w1))

        # ---------- Stage 3 ----------
        kv32_nlc = nchw_to_nlc(self.cat_key_d32([d14, d23, d22, e1]))
        d32 = nlc_to_nchw(self.blocks[2][2](e2_nlc, kv32_nlc, H_kv=h4, W_kv=w4, H_q=h2, W_q=w2), (h2, w2))

        kv31_nlc = nchw_to_nlc(self.cat_key_d31([d14, d23, d32, d21]))
        d31 = nlc_to_nchw(self.blocks[3][2](e1_nlc, kv31_nlc, H_kv=h4, W_kv=w4, H_q=h1, W_q=w1), (h1, w1))

        # ---------- Stage 4 ----------
        kv41_nlc = nchw_to_nlc(self.cat_key_d41([d14, d23, d32, d31]))
        d41 = nlc_to_nchw(self.blocks[3][3](e1_nlc, kv41_nlc, H_kv=h4, W_kv=w4, H_q=h1, W_q=w1), (h1, w1))

        # ---------- Output ----------
        if self.out_mode == 'd41':
            feat = d41
        else:
            # CatKey output is stage4 resolution; upsample to stage1 for prediction.
            feat = self.cat_key_fuse([d14, d13 + d23, d12 + d22 + d32, d11 + d21 + d31 + d41])  # [B, Csum, h4, w4]
            feat = resize(feat, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        if self.fuse is not None:
            feat = self.fuse(feat)

        seg_logits = self.cls_seg(feat)
        return seg_logits
