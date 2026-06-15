# Copyright (c) OpenMMLab. All rights reserved.
# This file implements an improved U-MixFormer-style decoder head with a
# novel Edge-enhanced Hybrid Cross-Agent Attention interaction module.

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init, trunc_normal_
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc


def _to_2tuple(x) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return int(x[0]), int(x[1])
    return int(x), int(x)


class BaseModuleInit(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            elif isinstance(m, nn.Parameter):
                trunc_normal_(m, std=0.02)


class DWConv(BaseModule):
    """Depth-wise conv used inside MixFFN (as in SegFormer/MixFFN style)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride=1, padding=pad, groups=dim, bias=True)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class MixFFN(BaseModule):
    """MLP + DWConv positional mixing (same spirit as MixFFN in SegFormer)."""

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        ffn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        feedforward_channels = int(feedforward_channels or embed_dims * 4)

        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.dwconv = DWConv(feedforward_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)
        self.drop = nn.Dropout(ffn_drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModule):
    """Pool multi-scale feature maps to the deepest resolution then concatenate.

    Args:
        pool_ratio: pooling ratios for [s4, s3, s2, s1] (deep -> shallow).
        dim: channel dims for [s4, s3, s2, s1] (deep -> shallow).
    """

    def __init__(self, pool_ratio: Sequence[int], dim: Sequence[int]):
        super().__init__()
        assert len(pool_ratio) == len(dim)
        self.pool_ratio = list(pool_ratio)

        self.sr_list = ModuleList()
        self.pool_list = ModuleList()
        for i, r in enumerate(self.pool_ratio):
            if r > 1:
                self.pool_list.append(nn.AvgPool2d(kernel_size=r, stride=r, ceil_mode=True))
                self.sr_list.append(nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1))

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        # feats: [c4, c3, c2, c1] in NCHW
        assert len(feats) == len(self.pool_ratio)
        outs: List[torch.Tensor] = []
        cnt = 0
        for i, r in enumerate(self.pool_ratio):
            x = feats[i]
            if r > 1:
                x = self.pool_list[cnt](x)
                x = self.sr_list[cnt](x)
                cnt += 1
            outs.append(x)
        return torch.cat(outs, dim=1)


class CrossPosBias(BaseModule):
    """A lightweight decomposed (H/W) cross-position bias for cross-attention.

    This is inspired by decomposed relative position techniques (e.g., ViTDet),
    but adapted to cross-attention between different spatial sizes (Hq,Wq) and
    (Hk,Wk) via interpolation.

    Bias for a query position (hq,wq) attending to key position (hk,wk):
        B(hq,wq,hk,wk) = Bh(hq,hk) + Bw(wq,wk)
    """

    def __init__(self, num_heads: int, base_size: int = 7):
        super().__init__()
        self.num_heads = int(num_heads)
        self.base_size = int(base_size)

        self.bias_h = nn.Parameter(torch.zeros(num_heads, base_size, base_size))
        self.bias_w = nn.Parameter(torch.zeros(num_heads, base_size, base_size))

    def get_bias(
        self,
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return Bh: (heads, Hq, Hk), Bw: (heads, Wq, Wk)
        bias_h = self.bias_h.unsqueeze(1).to(device=device, dtype=dtype)  # (heads,1,base,base)
        bias_w = self.bias_w.unsqueeze(1).to(device=device, dtype=dtype)
        bias_h = F.interpolate(bias_h, size=(Hq, Hk), mode='bilinear', align_corners=False).squeeze(1)
        bias_w = F.interpolate(bias_w, size=(Wq, Wk), mode='bilinear', align_corners=False).squeeze(1)
        return bias_h, bias_w


class CrossAttentionWithBias(BaseModule):
    """Cross-attention with optional decomposed cross-position bias."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
        pos_bias_base_size: int = 7,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q={dim_q} must be divisible by num_heads={num_heads}'
        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_bias: Optional[CrossPosBias] = None
        if use_pos_bias:
            self.pos_bias = CrossPosBias(num_heads=num_heads, base_size=pos_bias_base_size)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        Hk: int,
        Wk: int,
        Hq: int,
        Wq: int,
    ) -> torch.Tensor:
        # x_q: (B, Nq, Cq), x_kv: (B, Nk, Ckv)
        B, Nq, Cq = x_q.shape
        _, Nk, _ = x_kv.shape

        q = self.q(x_q).reshape(B, Nq, self.num_heads, Cq // self.num_heads).permute(0, 2, 1, 3)  # (B,h,Nq,d)
        kv = self.kv(x_kv).reshape(B, Nk, 2, self.num_heads, Cq // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,h,Nk,d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,h,Nq,Nk)

        if self.pos_bias is not None:
            bh, bw = self.pos_bias.get_bias(Hq, Wq, Hk, Wk, device=attn.device, dtype=attn.dtype)
            # reshape to (B,h,Hq,Wq,Hk,Wk) to add decomposed bias without allocating extra tensors
            attn = attn.view(B, self.num_heads, Hq, Wq, Hk, Wk)
            attn = attn + bh[None, :, :, None, :, None] + bw[None, :, None, :, None, :]
            attn = attn.view(B, self.num_heads, Nq, Nk)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class EdgeAwareAgentPool(BaseModule):
    """Edge-aware weighted pooling for generating boundary-sensitive agent tokens.

    Given projected query feature Q in NCHW, we build a high-frequency magnitude map:
        Hp = Q - AvgPool(Q, 3x3)
        m  = mean(|Hp|, channel)
        w  = sigmoid(m)

    Then compute weighted pooled tokens:
        A_edge = Pool(Q ⊙ w) / (Pool(w) + eps)
    """

    def __init__(self, out_shape: Tuple[int, int], eps: float = 1e-6):
        super().__init__()
        self.out_shape = _to_2tuple(out_shape)
        self.pool = nn.AdaptiveAvgPool2d(self.out_shape)
        self.eps = float(eps)

    def forward(self, q_map: torch.Tensor) -> torch.Tensor:
        # q_map: (B, C, H, W)
        lp = F.avg_pool2d(q_map, kernel_size=3, stride=1, padding=1)
        hp = q_map - lp
        mag = hp.abs().mean(dim=1, keepdim=True)  # (B,1,H,W)
        w = torch.sigmoid(mag)
        num = self.pool(q_map * w)
        den = self.pool(w)
        return num / (den + self.eps)


class AgentRelPosBias(BaseModule):
    """AgentAttention-style relative position bias, generalized to cross-attention.

    It produces two bias tensors:
      - B_ak: bias for agent_tokens -> key_tokens  (B, heads, Nag, Nk)
      - B_qa: bias for query_tokens -> agent_tokens (B, heads, Nq, Nag)

    This module follows the interpolation strategy in the official AgentAttention
    implementation (agent_attention.py), but allows Hq/Wq and Hk/Wk to differ.
    """

    def __init__(
        self,
        num_heads: int,
        agent_shape: Tuple[int, int],
        base_bias_size: int = 7,
        base_window_size: int = 7,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.agent_shape = _to_2tuple(agent_shape)
        assert self.agent_shape[0] == self.agent_shape[1], 'Only square agent_shape is supported for now.'
        self.pool_size = int(self.agent_shape[0])
        self.agent_num = int(self.agent_shape[0] * self.agent_shape[1])
        self.base_bias_size = int(base_bias_size)
        self.base_window_size = int(base_window_size)

        # Note: base_bias_size is fixed at 7 in the official AgentAttention.
        self.an_bias = nn.Parameter(torch.zeros(num_heads, self.agent_num, base_bias_size, base_bias_size))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, self.agent_num, base_bias_size, base_bias_size))

        # Window-size biases (factorized)
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, self.agent_num, base_window_size, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, self.agent_num, 1, base_window_size))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, base_window_size, 1, self.agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, base_window_size, self.agent_num))

        self.attn_drop = nn.Dropout(attn_drop)

    def _interp_bias_agent_to_spatial(self, bias: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # bias: (heads, agent_num, base, base) -> (heads, agent_num, H, W)
        return F.interpolate(bias, size=(H, W), mode='bilinear', align_corners=False)

    def _interp_bias_spatial_to_agent(self, bias_hw: torch.Tensor) -> torch.Tensor:
        # bias_hw: (heads, H*W, pool, pool) -> interpolate to agent_shape then reshape
        # Note: agent_shape == (pool,pool), so this is identity but kept for completeness.
        bias_hw = F.interpolate(bias_hw, size=self.agent_shape, mode='bilinear', align_corners=False)
        return bias_hw

    def get_bias(
        self,
        batch_size: int,
        Hk: int,
        Wk: int,
        Hq: int,
        Wq: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (bias_ak, bias_qa)."""
        heads = self.num_heads
        Nag = self.agent_num
        Nk = Hk * Wk
        Nq = Hq * Wq

        # -------- agent -> key (A->K) --------
        # position_bias1 from an_bias
        pb1 = self._interp_bias_agent_to_spatial(self.an_bias.to(device=device, dtype=dtype), Hk, Wk)
        # (heads, agent_num, Hk, Wk) -> (heads, pool, pool, Nk) -> (heads, Nk, pool, pool)
        pb1 = pb1.reshape(heads, self.pool_size, self.pool_size, Nk).permute(0, 3, 1, 2)
        pb1 = self._interp_bias_spatial_to_agent(pb1)
        # (heads, Nk, pool, pool) -> (heads, Nk, Nag) -> (heads, Nag, Nk)
        pb1 = pb1.reshape(heads, Nk, Nag).permute(0, 2, 1)
        pb1 = pb1.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B,heads,Nag,Nk)

        # position_bias2 from (ah_bias + aw_bias)
        pb2 = (self.ah_bias + self.aw_bias).squeeze(0).to(
            device=device, dtype=dtype,
        )  # (heads, Nag, baseW, baseW) as (heads,Nag,base,base)? Actually (heads,Nag,base,base) after interp below
        pb2 = F.interpolate(pb2, size=(Hk, Wk), mode='bilinear', align_corners=False)
        pb2 = pb2.reshape(heads, self.pool_size, self.pool_size, Nk).permute(0, 3, 1, 2)
        pb2 = self._interp_bias_spatial_to_agent(pb2)
        pb2 = pb2.reshape(heads, Nk, Nag).permute(0, 2, 1)
        pb2 = pb2.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        bias_ak = pb1 + pb2  # (B,heads,Nag,Nk)

        # -------- query -> agent (Q->A) --------
        # agent_bias1 from na_bias
        ab1 = self._interp_bias_agent_to_spatial(self.na_bias.to(device=device, dtype=dtype), Hq, Wq)
        ab1 = ab1.reshape(heads, self.pool_size, self.pool_size, Nq).permute(0, 3, 1, 2)
        ab1 = self._interp_bias_spatial_to_agent(ab1)
        ab1 = ab1.reshape(heads, Nq, Nag)
        ab1 = ab1.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B,heads,Nq,Nag)

        # agent_bias2 from (ha_bias + wa_bias)
        ab2 = (self.ha_bias + self.wa_bias).squeeze(0).to(device=device, dtype=dtype)  # (heads, base, base, Nag)
        ab2 = ab2.permute(0, 3, 1, 2)  # (heads, Nag, base, base)
        ab2 = F.interpolate(ab2, size=(Hq, Wq), mode='bilinear', align_corners=False)
        ab2 = ab2.reshape(heads, self.pool_size, self.pool_size, Nq).permute(0, 3, 1, 2)
        ab2 = self._interp_bias_spatial_to_agent(ab2)
        ab2 = ab2.reshape(heads, Nq, Nag)
        ab2 = ab2.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        bias_qa = ab1 + ab2  # (B,heads,Nq,Nag)

        return bias_ak, bias_qa


class AgentCrossAttention(BaseModule):
    """Cross AgentAttention with optional edge-aware agent tokens.

    It computes:
      Out = AgentAttn_sem(Q, K, V) + AgentAttn_edge(Q, K, V)   (optional edge)
    where AgentAttn follows the 2-step agent attention (agent->key then query->agent).

    Note:
      - We keep additive fusion (no alpha-gating) as suggested by empirical findings.
      - For cross setting, K/V are projected from mixed key feature, while agent tokens
        are pooled from projected Q at query resolution.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        agent_shape: Tuple[int, int] = (7, 7),
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
        pos_bias_base_size: int = 7,
        pos_bias_base_window_size: int = 7,
        use_edge_agent: bool = True,
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f'dim_q={dim_q} must be divisible by num_heads={num_heads}'
        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        self.agent_shape = _to_2tuple(agent_shape)
        self.agent_num = int(self.agent_shape[0] * self.agent_shape[1])
        self.pool_sem = nn.AdaptiveAvgPool2d(self.agent_shape)

        self.edge_pool: Optional[EdgeAwareAgentPool] = None
        if use_edge_agent:
            self.edge_pool = EdgeAwareAgentPool(self.agent_shape)

        self.pos_bias: Optional[AgentRelPosBias] = None
        if use_pos_bias:
            self.pos_bias = AgentRelPosBias(
                num_heads=num_heads,
                agent_shape=self.agent_shape,
                base_bias_size=pos_bias_base_size,
                base_window_size=pos_bias_base_window_size,
                attn_drop=attn_drop,
            )

        self.dwc: Optional[nn.Conv2d] = None
        if use_dwc:
            pad = dwc_kernel_size // 2
            self.dwc = nn.Conv2d(dim_q, dim_q, kernel_size=dwc_kernel_size, padding=pad, groups=dim_q)

        if self.dwc is not None:
            # Kaiming init for conv
            fan_out = self.dwc.kernel_size[0] * self.dwc.kernel_size[1] * self.dwc.out_channels
            fan_out //= self.dwc.groups
            self.dwc.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if self.dwc.bias is not None:
                self.dwc.bias.data.zero_()

    def _run_agent_attn(
        self,
        q: torch.Tensor,  # (B,h,Nq,d)
        k: torch.Tensor,  # (B,h,Nk,d)
        v: torch.Tensor,  # (B,h,Nk,d)
        agent_tokens: torch.Tensor,  # (B,h,Nag,d)
        bias_ak: Optional[torch.Tensor],  # (B,h,Nag,Nk)
        bias_qa: Optional[torch.Tensor],  # (B,h,Nq,Nag)
    ) -> torch.Tensor:
        # agent -> key
        attn1 = (agent_tokens * self.scale) @ k.transpose(-2, -1)  # (B,h,Nag,Nk)
        if bias_ak is not None:
            attn1 = attn1 + bias_ak
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.attn_drop(attn1)
        agent_v = attn1 @ v  # (B,h,Nag,d)

        # query -> agent
        attn2 = (q * self.scale) @ agent_tokens.transpose(-2, -1)  # (B,h,Nq,Nag)
        if bias_qa is not None:
            attn2 = attn2 + bias_qa
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.attn_drop(attn2)
        out = attn2 @ agent_v  # (B,h,Nq,d)
        return out

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        Hk: int,
        Wk: int,
        Hq: int,
        Wq: int,
    ) -> torch.Tensor:
        # x_q: (B,Nq,Cq), x_kv: (B,Nk,Ckv)
        B, Nq, Cq = x_q.shape
        _, Nk, _ = x_kv.shape
        head_dim = Cq // self.num_heads

        q_lin = self.q(x_q)  # (B,Nq,Cq)
        q = q_lin.reshape(B, Nq, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B,h,Nq,d)

        kv = self.kv(x_kv).reshape(B, Nk, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,h,Nk,d)

        # agent tokens from projected query map
        q_map = q_lin.transpose(1, 2).reshape(B, Cq, Hq, Wq)  # (B,C,Hq,Wq)
        a_sem = self.pool_sem(q_map).flatten(2).transpose(1, 2)  # (B,Nag,C)
        a_sem = a_sem.reshape(B, self.agent_num, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B,h,Nag,d)

        a_edge: Optional[torch.Tensor] = None
        if self.edge_pool is not None:
            a_edge_map = self.edge_pool(q_map)  # (B,C,Ph,Pw)
            a_edge = a_edge_map.flatten(2).transpose(1, 2)
            a_edge = a_edge.reshape(B, self.agent_num, self.num_heads, head_dim).permute(0, 2, 1, 3)

        bias_ak: Optional[torch.Tensor] = None
        bias_qa: Optional[torch.Tensor] = None
        if self.pos_bias is not None:
            bias_ak, bias_qa = self.pos_bias.get_bias(
                batch_size=B,
                Hk=Hk,
                Wk=Wk,
                Hq=Hq,
                Wq=Wq,
                device=q.device,
                dtype=q.dtype,
            )

        out_sem = self._run_agent_attn(q, k, v, a_sem, bias_ak, bias_qa)

        out = out_sem
        if a_edge is not None:
            out_edge = self._run_agent_attn(q, k, v, a_edge, bias_ak, bias_qa)
            out = out + out_edge

        out = out.transpose(1, 2).reshape(B, Nq, Cq)  # (B,Nq,Cq)

        # diversity-preserving local residual (optional):
        # we upsample value map to query resolution then apply depthwise conv.
        if self.dwc is not None:
            v_map = v.transpose(1, 2).reshape(B, Nk, Cq)
            v_map = v_map.transpose(1, 2).reshape(B, Cq, Hk, Wk)
            if (Hk != Hq) or (Wk != Wq):
                v_map = F.interpolate(v_map, size=(Hq, Wq), mode='bilinear', align_corners=False)
            out = out + self.dwc(v_map).flatten(2).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class EdgeHybridInteraction(BaseModule):
    """Edge-enhanced Hybrid Cross-Agent interaction.

    Out = CrossAttn(Q, KV) + AgentCrossAttn(Q, KV)

    This is a direct additive fusion to preserve complementary benefits and
    avoid alpha-gating degradation.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        # Cross branch
        use_cross: bool = True,
        use_cross_pos_bias: bool = True,
        cross_pos_bias_base_size: int = 7,
        # Agent branch
        use_agent: bool = True,
        agent_shape: Tuple[int, int] = (7, 7),
        use_agent_pos_bias: bool = True,
        agent_pos_bias_base_size: int = 7,
        agent_pos_bias_base_window_size: int = 7,
        use_edge_agent: bool = True,
        use_agent_dwc: bool = True,
        agent_dwc_kernel_size: int = 3,
    ):
        super().__init__()
        self.use_cross = bool(use_cross)
        self.use_agent = bool(use_agent)

        self.cross_attn: Optional[CrossAttentionWithBias] = None
        if self.use_cross:
            self.cross_attn = CrossAttentionWithBias(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_pos_bias=use_cross_pos_bias,
                pos_bias_base_size=cross_pos_bias_base_size,
            )

        self.agent_attn: Optional[AgentCrossAttention] = None
        if self.use_agent:
            self.agent_attn = AgentCrossAttention(
                dim_q=dim_q,
                dim_kv=dim_kv,
                num_heads=num_heads,
                agent_shape=agent_shape,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_pos_bias=use_agent_pos_bias,
                pos_bias_base_size=agent_pos_bias_base_size,
                pos_bias_base_window_size=agent_pos_bias_base_window_size,
                use_edge_agent=use_edge_agent,
                use_dwc=use_agent_dwc,
                dwc_kernel_size=agent_dwc_kernel_size,
            )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        Hk: int,
        Wk: int,
        Hq: int,
        Wq: int,
    ) -> torch.Tensor:
        out = 0.0
        if self.cross_attn is not None:
            out = out + self.cross_attn(x_q, x_kv, Hk=Hk, Wk=Wk, Hq=Hq, Wq=Wq)
        if self.agent_attn is not None:
            out = out + self.agent_attn(x_q, x_kv, Hk=Hk, Wk=Wk, Hq=Hq, Wq=Wq)
        return out


class EdgeHybridBlock(BaseModule):
    """A transformer block used in the decoder stages."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        # interaction config
        use_cross: bool = True,
        use_cross_pos_bias: bool = True,
        cross_pos_bias_base_size: int = 7,
        use_agent: bool = True,
        agent_shape: Tuple[int, int] = (7, 7),
        use_agent_pos_bias: bool = True,
        agent_pos_bias_base_size: int = 7,
        agent_pos_bias_base_window_size: int = 7,
        use_edge_agent: bool = True,
        use_agent_dwc: bool = True,
        agent_dwc_kernel_size: int = 3,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.interaction = EdgeHybridInteraction(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cross=use_cross,
            use_cross_pos_bias=use_cross_pos_bias,
            cross_pos_bias_base_size=cross_pos_bias_base_size,
            use_agent=use_agent,
            agent_shape=agent_shape,
            use_agent_pos_bias=use_agent_pos_bias,
            agent_pos_bias_base_size=agent_pos_bias_base_size,
            agent_pos_bias_base_window_size=agent_pos_bias_base_window_size,
            use_edge_agent=use_edge_agent,
            use_agent_dwc=use_agent_dwc,
            agent_dwc_kernel_size=agent_dwc_kernel_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim_q)
        self.ffn = MixFFN(embed_dims=dim_q, feedforward_channels=mlp_hidden_dim, ffn_drop=drop)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor, Hk: int, Wk: int, Hq: int, Wq: int) -> torch.Tensor:
        # Pre-norm
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)
        x_q = x_q + self.drop_path(self.interaction(q, kv, Hk=Hk, Wk=Wk, Hq=Hq, Wq=Wq))
        x_q = x_q + self.drop_path(self.ffn(self.norm_ffn(x_q), H=Hq, W=Wq))
        return x_q


@MODELS.register_module()
class EHCAHead(BaseDecodeHead):
    """Improved U-MixFormer-style decoder head with Edge-Hybrid interaction.

    Key differences vs APFormerHead2 (official U-MixFormer head):
      1) Replace plain CrossAttention with EdgeHybridInteraction:
         - Cross-attn + Agent cross-attn (with agent relative position bias)
         - Add a decomposed cross-position bias on the cross-attn branch
         - Optional edge-aware agent tokens for boundary-sensitive context

    Args:
        num_heads: attention heads for stages [s4,s3,s2,s1].
        pool_ratio: CatKey pooling ratios for [s4,s3,s2,s1].
        agent_shape: agent token grids for [s4,s3,s2,s1] (each is (P,P)).
        use_edge_agent: whether to enable edge-aware agent tokens at each stage.
        use_cross_pos_bias: enable decomposed cross-position bias for cross-attn.
        use_agent: enable agent cross-attn branch.
        use_cross: enable cross-attn branch.
    """

    def __init__(
        self,
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        # block hyperparams
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        # hybrid interaction switches
        use_cross: bool = True,
        use_agent: bool = True,
        # cross position bias
        use_cross_pos_bias: bool = True,
        cross_pos_bias_base_size: int = 7,
        # agent relative bias
        use_agent_pos_bias: bool = True,
        agent_pos_bias_base_size: int = 7,
        agent_pos_bias_base_window_size: int = 7,
        # agent shapes & edge agents per stage
        agent_shape: Sequence[Tuple[int, int]] = ((5, 5), (5, 5), (7, 7), (7, 7)),
        use_edge_agent: Sequence[bool] = (False, False, True, True),
        use_agent_dwc: bool = True,
        agent_dwc_kernel_size: int = 3,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(num_heads) == 4, 'num_heads must have 4 elements for [s4,s3,s2,s1].'
        assert len(pool_ratio) == 4, 'pool_ratio must have 4 elements for [s4,s3,s2,s1].'
        assert len(agent_shape) == 4, 'agent_shape must have 4 elements for [s4,s3,s2,s1].'
        assert len(use_edge_agent) == 4, 'use_edge_agent must have 4 elements for [s4,s3,s2,s1].'

        # in_channels order in mmseg is typically [c1,c2,c3,c4] (shallow->deep)
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = int(c1_in + c2_in + c3_in + c4_in)

        # CatKey modules (progressive key update like APFormerHead2)
        # Each CatKey pools [c4,c3,c2,c1] to resolution of c4.
        self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=(c4_in, c3_in, c2_in, c1_in))
        self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=(c4_in, c3_in, c2_in, c1_in))

        # DropPath schedule: simple constant or linear across blocks.
        # We use a short linear schedule across 4 stages for stability.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        # Stage blocks (s4->s1)
        self.block_s4 = EdgeHybridBlock(
            dim_q=c4_in,
            dim_kv=tot_channels,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[0],
            use_cross=use_cross,
            use_cross_pos_bias=use_cross_pos_bias,
            cross_pos_bias_base_size=cross_pos_bias_base_size,
            use_agent=use_agent,
            agent_shape=agent_shape[0],
            use_agent_pos_bias=use_agent_pos_bias,
            agent_pos_bias_base_size=agent_pos_bias_base_size,
            agent_pos_bias_base_window_size=agent_pos_bias_base_window_size,
            use_edge_agent=use_edge_agent[0],
            use_agent_dwc=use_agent_dwc,
            agent_dwc_kernel_size=agent_dwc_kernel_size,
        )

        self.block_s3 = EdgeHybridBlock(
            dim_q=c3_in,
            dim_kv=tot_channels,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[1],
            use_cross=use_cross,
            use_cross_pos_bias=use_cross_pos_bias,
            cross_pos_bias_base_size=cross_pos_bias_base_size,
            use_agent=use_agent,
            agent_shape=agent_shape[1],
            use_agent_pos_bias=use_agent_pos_bias,
            agent_pos_bias_base_size=agent_pos_bias_base_size,
            agent_pos_bias_base_window_size=agent_pos_bias_base_window_size,
            use_edge_agent=use_edge_agent[1],
            use_agent_dwc=use_agent_dwc,
            agent_dwc_kernel_size=agent_dwc_kernel_size,
        )

        self.block_s2 = EdgeHybridBlock(
            dim_q=c2_in,
            dim_kv=tot_channels,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[2],
            use_cross=use_cross,
            use_cross_pos_bias=use_cross_pos_bias,
            cross_pos_bias_base_size=cross_pos_bias_base_size,
            use_agent=use_agent,
            agent_shape=agent_shape[2],
            use_agent_pos_bias=use_agent_pos_bias,
            agent_pos_bias_base_size=agent_pos_bias_base_size,
            agent_pos_bias_base_window_size=agent_pos_bias_base_window_size,
            use_edge_agent=use_edge_agent[2],
            use_agent_dwc=use_agent_dwc,
            agent_dwc_kernel_size=agent_dwc_kernel_size,
        )

        self.block_s1 = EdgeHybridBlock(
            dim_q=c1_in,
            dim_kv=tot_channels,
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[3],
            use_cross=use_cross,
            use_cross_pos_bias=use_cross_pos_bias,
            cross_pos_bias_base_size=cross_pos_bias_base_size,
            use_agent=use_agent,
            agent_shape=agent_shape[3],
            use_agent_pos_bias=use_agent_pos_bias,
            agent_pos_bias_base_size=agent_pos_bias_base_size,
            agent_pos_bias_base_window_size=agent_pos_bias_base_window_size,
            use_edge_agent=use_edge_agent[3],
            use_agent_dwc=use_agent_dwc,
            agent_dwc_kernel_size=agent_dwc_kernel_size,
        )

        # Fuse multi-stage outputs to channels
        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=int(self.channels),
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        # inputs order in mmseg with multiple_select: [c1,c2,c3,c4] (shallow->deep)
        feats = self._transform_inputs(inputs)
        c1, c2, c3, c4 = feats

        B, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # -------- stage s4 --------
        key = self.cat_key1([c4, c3, c2, c1])  # (B, sumC, h4, w4)
        key_tokens = key.flatten(2).transpose(1, 2)  # (B, Nk, sumC)

        q4 = c4.flatten(2).transpose(1, 2)  # (B, N4, C4)
        out4 = self.block_s4(q4, key_tokens, Hk=h4, Wk=w4, Hq=h4, Wq=w4)
        out4_map = out4.transpose(1, 2).reshape(B, -1, h4, w4)

        # -------- stage s3 --------
        key = self.cat_key2([out4_map, c3, c2, c1])
        key_tokens = key.flatten(2).transpose(1, 2)

        q3 = c3.flatten(2).transpose(1, 2)
        out3 = self.block_s3(q3, key_tokens, Hk=h4, Wk=w4, Hq=h3, Wq=w3)
        out3_map = out3.transpose(1, 2).reshape(B, -1, h3, w3)

        # -------- stage s2 --------
        key = self.cat_key3([out4_map, out3_map, c2, c1])
        key_tokens = key.flatten(2).transpose(1, 2)

        q2 = c2.flatten(2).transpose(1, 2)
        out2 = self.block_s2(q2, key_tokens, Hk=h4, Wk=w4, Hq=h2, Wq=w2)
        out2_map = out2.transpose(1, 2).reshape(B, -1, h2, w2)

        # -------- stage s1 --------
        key = self.cat_key4([out4_map, out3_map, out2_map, c1])
        key_tokens = key.flatten(2).transpose(1, 2)

        q1 = c1.flatten(2).transpose(1, 2)
        out1 = self.block_s1(q1, key_tokens, Hk=h4, Wk=w4, Hq=h1, Wq=w1)
        out1_map = out1.transpose(1, 2).reshape(B, -1, h1, w1)

        # upsample to 1/4 resolution and fuse
        out4_up = resize(out4_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        out3_up = resize(out3_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        out2_up = resize(out2_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        fused = self.linear_fuse(torch.cat([out4_up, out3_up, out2_up, out1_map], dim=1))

        # Use BaseDecodeHead classifier (dropout + conv_seg)
        seg_logits = self.cls_seg(fused)
        return seg_logits
