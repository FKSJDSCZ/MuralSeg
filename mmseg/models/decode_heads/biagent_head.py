# A novel decoder head based on U-MixFormer + Agent Attention style cross attention.
# This file is intended to be placed under: mmseg/models/decode_heads/
#
# Key ideas:
# 1) Replace vanilla CrossAttention in U-MixFormer with Cross-Agent-Attention (CAA).
# 2) Add an iterative bidirectional interaction per stage (forward + feedback).
# 3) Provide edge-aware agent token generation and cross-scale positional bias options.
#
# NOTE: This implementation follows mmsegmentation / mmengine / mmcv coding style.
#       All ablation knobs are exposed as __init__ arguments (no nested dicts).

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init, trunc_normal_
from mmengine.utils import to_2tuple, to_4tuple
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.drop import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc


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


class DWConv(BaseModuleInit):
    """Depth-wise conv used inside MLP (token -> featuremap -> token)."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, groups=dim, bias=True,
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModuleInit):
    """MLP with a depth-wise conv in between (same as many segmentation transformers)."""

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

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Downsample(BaseModuleInit):
    def __init__(self, pool_ratio: int, dim: int):
        super().__init__()
        if pool_ratio > 1:
            self.downsample = Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True),
                nn.AvgPool2d(pool_ratio, pool_ratio, ceil_mode=True),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class DirectionalDilatedPU(BaseModuleInit):
    """Directional Dilated Perception Unit, depthwise for efficiency.

    This module is inspired by SCASeg's Local Perception Module (LPM),
    but adds multi-dilation and optional directional kernels.
    """

    def __init__(
        self,
        dim: int,
        dilations=(1, 3, 5),
        use_directional=True,
        directional_kernel=7,
        se_reduction=4,
        act_layer=nn.GELU,
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

        # Channel gate (SE-style)
        mid = max(dim // int(se_reduction), 4)
        self.se = Sequential(
            nn.Conv2d(dim, mid, kernel_size=1, bias=True),
            act_layer(),
            nn.Conv2d(mid, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, Hq, Wq) -> torch.Tensor:
        feat = nlc_to_nchw(x, (Hq, Wq))
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


class BoundaryPrior(BaseModuleInit):
    """
    Lightweight boundary/structure prior from shallow feature (usually c1).
    Output is a single-channel weight map in [0,1] for edge-aware agent pooling.

    Design goal: provide a learnable 'high-frequency emphasis' WITHOUT FFT,
    cheap enough to compute once and re-use across stages.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 32,
        norm_cfg: Optional[dict] = dict(type='SyncBN', requires_grad=True),
    ):
        super().__init__()
        self.reduce = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='GELU'),
        )
        # depth-wise conv to capture local gradients / edges
        self.dw = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=1,
            padding=1, groups=mid_channels, bias=True,
        )
        self.proj = nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.reduce(x)
        x = self.dw(x)
        x = self.proj(x)
        return torch.sigmoid(x)


class AgentTokenGenerator(BaseModuleInit):
    """
    Generate agent tokens A from query tokens (optionally edge-aware).
    Supports:
        - 'avgpool'   : vanilla AdaptiveAvgPool2d
        - 'edge_pool' : weighted pooling using a boundary prior map
        - 'learnable' : purely learnable agent tokens (like object queries)
        - 'hybrid'    : learnable seeds + pooled dynamic residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        agent_shape: Tuple[int, int] = (7, 7),
        agent_token_type: str = 'avgpool',
        eps: float = 1e-4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        assert self.dim % self.num_heads == 0
        self.head_dim = self.dim // self.num_heads

        self.agent_shape: Tuple[int, int] = to_2tuple(agent_shape)
        self.agent_num = self.agent_shape[0] * self.agent_shape[1]
        self.agent_token_type = str(agent_token_type)
        self.eps = float(eps)

        if self.agent_token_type in ('avgpool', 'edge_pool', 'hybrid'):
            self.pool = nn.AdaptiveAvgPool2d(self.agent_shape)
        else:
            self.pool = None

        if self.agent_token_type in ('learnable', 'hybrid'):
            # Learnable seed tokens per-head (1, heads, Na, head_dim)
            self.agent_seeds = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, self.head_dim))
        else:
            self.agent_seeds = None

        if self.agent_token_type not in ('avgpool', 'edge_pool', 'learnable', 'hybrid'):
            raise ValueError(f'Unsupported agent_token_type={self.agent_token_type}')

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) query tokens (typically already normalized).
            H, W: spatial size for tokens (N == H*W).
            weight_map: (B, 1, H, W) in [0,1], used only for edge_pool/hybrid.
        Returns:
            agent_tokens: (B, heads, Na, head_dim)
        """
        B, N, C = x.shape
        assert N == H * W, f'N ({N}) must equal H*W ({H}*{W})'
        x_map = x.transpose(1, 2).reshape(B, C, H, W)

        if self.agent_token_type == 'learnable':
            # Pure learnable queries: expand to batch.
            return self.agent_seeds.expand(B, -1, -1, -1)

        # pooled dynamic agents
        if self.agent_token_type in ('edge_pool', 'hybrid'):
            assert weight_map is not None, 'edge_pool/hybrid requires weight_map'
            # Weighted pooling: pool(x*w) / (pool(w)+eps)
            num = self.pool(x_map * weight_map)
            den = self.pool(weight_map) + self.eps
            agent_map = num / den
        else:
            # avgpool
            agent_map = self.pool(x_map)

        agent = agent_map.flatten(2).transpose(1, 2)  # (B, Na, C)
        agent = agent.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.agent_token_type == 'hybrid':
            agent = agent + self.agent_seeds.expand(B, -1, -1, -1)

        return agent


class InterpolatedDecomposedBias(BaseModuleInit):
    """
    Interpolated decomposed bias (in the spirit of official AgentAttention bias),
    but generalized to arbitrary rectangular agent grids and cross-scale attention.

    It produces:
        - bias_a2k: (B, heads, Na, Nk)
        - bias_q2a: (B, heads, Nq, Na)
    """

    def __init__(
        self,
        num_heads: int,
        agent_shape: Tuple[int, int],
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.agent_shape = to_2tuple(agent_shape)
        self.agent_num = self.agent_shape[0] * self.agent_shape[1]

        ah, aw = self.agent_shape

        # Following the official form: two 2D tables + two axial (H/W) tables, per direction.
        # Agent -> token (a2k)
        self.an_bias = nn.Parameter(torch.zeros(self.num_heads, self.agent_num, ah, aw))
        self.ah_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, ah, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, 1, aw))
        # Token -> agent (q2a)
        self.na_bias = nn.Parameter(torch.zeros(self.num_heads, self.agent_num, ah, aw))
        self.ha_bias = nn.Parameter(torch.zeros(1, self.num_heads, ah, 1, self.agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, aw, self.agent_num))

    def forward(
        self,
        B: int,
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bias_a2k: (B, heads, Na, Nk)
            bias_q2a: (B, heads, Nq, Na)
        """
        Nk = Hk * Wk
        Nq = Hq * Wq

        # -------------------------
        # bias for agent -> key/tokens
        # -------------------------
        # an_bias: (heads, Na, ah, aw) -> interpolate to (Hk, Wk)
        bias1 = F.interpolate(self.an_bias, size=(Hk, Wk), mode='bilinear', align_corners=False)
        bias1 = bias1.flatten(2)  # (heads, Na, Nk)

        bias2 = (self.ah_bias + self.aw_bias).squeeze(0)  # (heads, Na, ah, aw)
        bias2 = F.interpolate(bias2, size=(Hk, Wk), mode='bilinear', align_corners=False)
        bias2 = bias2.flatten(2)  # (heads, Na, Nk)

        bias_a2k = (bias1 + bias2).unsqueeze(0).expand(B, -1, -1, -1).to(device)

        # -------------------------
        # bias for query/tokens -> agent
        # -------------------------
        bias3 = F.interpolate(self.na_bias, size=(Hq, Wq), mode='bilinear', align_corners=False)
        bias3 = bias3.flatten(2).permute(0, 2, 1)  # (heads, Nq, Na)

        bias4 = (self.ha_bias + self.wa_bias).squeeze(0)  # (heads, ah, aw, Na)
        bias4 = bias4.permute(0, 3, 1, 2)  # (heads, Na, ah, aw)
        bias4 = F.interpolate(bias4, size=(Hq, Wq), mode='bilinear', align_corners=False)
        bias4 = bias4.flatten(2).permute(0, 2, 1)  # (heads, Nq, Na)

        bias_q2a = (bias3 + bias4).unsqueeze(0).expand(B, -1, -1, -1).to(device)

        return bias_a2k, bias_q2a


class ContinuousRelativeBias(BaseModuleInit):
    """
    Continuous relative position bias (SwinV2-style idea) extended to
    *agent-token* interactions and cross-scale attention.

    It avoids hand-crafted interpolation of bias tables and is naturally
    compatible with varying (Hq,Wq) and (Hk,Wk) across decoder stages.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)

        # Two MLPs: agent->key and query->agent
        self.mlp_a2k = Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        self.mlp_q2a = Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_heads),
        )

    @staticmethod
    def _make_coords(H: int, W: int, device: torch.device) -> torch.Tensor:
        """Return normalized coords in [-1,1] with shape (H*W, 2)."""
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(H * W, 2)  # (N,2)
        return coords

    @staticmethod
    def _make_agent_coords(agent_shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Agent grid coords in [-1,1], shape (Na,2)."""
        ah, aw = agent_shape
        ys = torch.linspace(-1.0, 1.0, ah, device=device)
        xs = torch.linspace(-1.0, 1.0, aw, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(ah * aw, 2)
        return coords

    def forward(
        self,
        B: int,
        agent_shape: Tuple[int, int],
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            bias_a2k: (B, heads, Na, Nk)
            bias_q2a: (B, heads, Nq, Na)
        """
        agent_shape = to_2tuple(agent_shape)
        Na = agent_shape[0] * agent_shape[1]
        Nk = Hk * Wk
        Nq = Hq * Wq

        coords_a = self._make_agent_coords(agent_shape, device)  # (Na,2)
        coords_k = self._make_coords(Hk, Wk, device)  # (Nk,2)
        coords_q = self._make_coords(Hq, Wq, device)  # (Nq,2)

        # scale features (log ratios)
        s_h = math.log((Hk + 1e-6) / (Hq + 1e-6))
        s_w = math.log((Wk + 1e-6) / (Wq + 1e-6))
        scale_feat_a2k = torch.tensor([s_h, s_w], device=device).view(1, 1, 2)  # (1,1,2)
        scale_feat_q2a = torch.tensor([-s_h, -s_w], device=device).view(1, 1, 2)

        # Agent -> Key bias
        rel_a2k = coords_a[:, None, :] - coords_k[None, :, :]  # (Na, Nk, 2)
        rel_a2k = torch.cat([rel_a2k, scale_feat_a2k.expand(Na, Nk, 2)], dim=-1)  # (Na,Nk,4)
        bias_a2k = self.mlp_a2k(rel_a2k).permute(2, 0, 1)  # (heads, Na, Nk)
        bias_a2k = bias_a2k.unsqueeze(0).expand(B, -1, -1, -1)

        # Query -> Agent bias
        rel_q2a = coords_q[:, None, :] - coords_a[None, :, :]  # (Nq, Na, 2)
        rel_q2a = torch.cat([rel_q2a, scale_feat_q2a.expand(Nq, Na, 2)], dim=-1)  # (Nq,Na,4)
        bias_q2a = self.mlp_q2a(rel_q2a).permute(2, 0, 1)  # (heads, Nq, Na)
        bias_q2a = bias_q2a.unsqueeze(0).expand(B, -1, -1, -1)

        return bias_a2k, bias_q2a


class CrossAgentAttention(BaseModuleInit):
    """
    Cross-Agent-Attention (CAA): cross-attention where interactions are mediated
    by a small set of agent tokens A.

    Output shape matches query tokens.
    """

    def __init__(
        self,
        input_dim_q: int,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        agent_shape: Tuple[int, int] = (7, 7),
        agent_token_type: str = 'avgpool',
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale: float = -0.5,
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim_q = int(input_dim_q)
        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        assert self.dim_q % self.num_heads == 0

        self.head_dim = self.dim_q // self.num_heads
        self.scale = self.head_dim ** scale

        self.q = nn.Linear(self.input_dim_q, self.dim_q, bias=qkv_bias)
        self.kv = nn.Linear(self.dim_kv, self.dim_q * 2, bias=qkv_bias)

        # Agent generator
        self.agent_generator = AgentTokenGenerator(
            dim=self.dim_q,
            num_heads=self.num_heads,
            agent_shape=agent_shape,
            agent_token_type=agent_token_type,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_q, self.dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_dwc = bool(use_dwc)
        if self.use_dwc:
            k = int(dwc_kernel_size)
            self.dwc = nn.Conv2d(
                self.dim_q, self.dim_q, kernel_size=k, stride=1,
                padding=k // 2, groups=self.dim_q, bias=True,
            )
        else:
            self.dwc = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        bias_a2k: Optional[torch.Tensor] = None,
        bias_q2a: Optional[torch.Tensor] = None,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_q: (B, Nq, Cq)
            x_kv: (B, Nk, Ckv)
            Hq, Wq: query spatial size
            Hk, Wk: key/value spatial size
            bias_a2k: (B, heads, Na, Nk)
            bias_q2a: (B, heads, Nq, Na)
            weight_map: (B, 1, H, W) in [0,1], used only for edge_pool/hybrid.
        Returns:
            out: (B, Nq, dim_q)
        """
        B, Nq, Cq = x_q.shape
        Nk = x_kv.shape[1]
        assert Nq == Hq * Wq, f'Nq mismatch: {Nq} vs {Hq}*{Wq}'
        assert Nk == Hk * Wk, f'Nk mismatch: {Nk} vs {Hk}*{Wk}'
        assert Cq == self.input_dim_q

        q = self.q(x_q)  # (B,Nq,C)
        kv = self.kv(x_kv).reshape(B, Nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,heads,Nk,d)
        agent_tokens = self.agent_generator(q, Hq, Wq, weight_map)

        q = q.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,heads,Nq,d)

        # ---------- agent -> key ----------
        # (B,heads,Na,Nk)
        a2k = (agent_tokens * self.scale) @ k.transpose(-2, -1)
        if bias_a2k is not None:
            a2k = a2k + bias_a2k
        a2k = self.softmax(a2k)
        a2k = self.attn_drop(a2k)
        agent_v = a2k @ v  # (B,heads,Na,d)

        # ---------- query -> agent ----------
        q2a = (q * self.scale) @ agent_tokens.transpose(-2, -1)  # (B,heads,Nq,Na)
        if bias_q2a is not None:
            q2a = q2a + bias_q2a
        q2a = self.softmax(q2a)
        q2a = self.attn_drop(q2a)

        out = q2a @ agent_v  # (B,heads,Nq,d)
        out = out.transpose(1, 2).reshape(B, Nq, self.dim_q)

        # Optional local enhancement (project V to query resolution)
        if self.use_dwc:
            v_map = nlc_to_nchw(v.transpose(1, 2).reshape(B, Nk, self.dim_q), (Hk, Wk))
            if (Hk, Wk) != (Hq, Wq):
                v_map = F.interpolate(v_map, size=(Hq, Wq), mode='bilinear', align_corners=False)
            out = out + nchw_to_nlc(self.dwc(v_map))

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class BiAgentBlock(BaseModuleInit):
    """
    A stage block that produces two feature maps via:
        1) forward cross-agent-attention: Q=encoder stage, KV=mixed cat_key
        2) feedback cross-agent-attention: Q=out1, KV=encoder stage feature
    """

    def __init__(
        self,
        enc_dim: int,
        feats_dim: int,
        num_heads: int,
        agent_shape: Tuple[int, int] = (7, 7),
        agent_token_type: str = 'avgpool',
        bias_type: str = 'crpb',
        crpb_hidden_dim: int = 128,
        mlp_ratio: Tuple[float, float] = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = True,
        use_bidirectional: bool = True,
        share_agent_tokens: bool = False,
        feedback_q_concat: bool = False,
        use_ddpu: bool = False,
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
    ):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.feats_dim = int(feats_dim)
        self.num_heads = int(num_heads)

        self.use_bidirectional = bool(use_bidirectional)
        self.share_agent_tokens = bool(share_agent_tokens)
        self.feedback_q_concat = bool(feedback_q_concat)
        self.use_ddpu = bool(use_ddpu)

        # Token norms
        self.norm_q1 = nn.LayerNorm(self.enc_dim)
        self.norm_kv1 = nn.LayerNorm(self.feats_dim)

        # Bias module (shared between the two passes inside the stage)
        bias_type = str(bias_type).lower()
        if bias_type == 'none':
            self.bias = None
            self.bias_type = 'none'
        elif bias_type in ('interp', 'interpolate', 'idb'):
            self.bias = InterpolatedDecomposedBias(num_heads=self.num_heads, agent_shape=agent_shape)
            self.bias_type = 'interp'
        elif bias_type in ('crpb', 'continuous'):
            self.bias = ContinuousRelativeBias(num_heads=self.num_heads, hidden_dim=crpb_hidden_dim)
            self.bias_type = 'crpb'
        else:
            raise ValueError(f'Unsupported bias_type={bias_type}')

        # Forward attention
        self.attn_fwd = CrossAgentAttention(
            input_dim_q=self.enc_dim,
            dim_q=self.enc_dim,
            dim_kv=self.feats_dim,
            num_heads=self.num_heads,
            agent_shape=agent_shape,
            agent_token_type=agent_token_type,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_dwc=use_dwc,
            dwc_kernel_size=dwc_kernel_size,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # FFN & norms
        if self.use_bidirectional:
            self.norm_mlp2 = nn.LayerNorm(self.enc_dim)
            self.mlp2 = Mlp(in_features=self.enc_dim, hidden_features=int(self.enc_dim * mlp_ratio[1]), drop=drop)
        self.norm_mlp1 = nn.LayerNorm(self.enc_dim)
        self.mlp1 = Mlp(in_features=self.enc_dim, hidden_features=int(self.enc_dim * mlp_ratio[0]), drop=drop)

        # DDPU
        if self.use_ddpu:
            self.norm_ddpu = nn.LayerNorm(self.enc_dim)
            self.ddpu = DirectionalDilatedPU(self.enc_dim)

        # Optional feedback pass
        if self.use_bidirectional:
            self.norm_q2 = nn.LayerNorm(self.enc_dim * 2)
            self.norm_kv2 = nn.LayerNorm(self.feats_dim)

            # if self.feedback_q_concat:
            #     self.feedback_q_proj = nn.Linear(self.enc_dim * 2, self.enc_dim, bias=True)

            self.attn_bwd = CrossAgentAttention(
                input_dim_q=self.enc_dim * 2,
                dim_q=self.enc_dim,
                dim_kv=self.feats_dim,
                num_heads=self.num_heads,
                agent_shape=agent_shape,
                agent_token_type=agent_token_type,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_dwc=use_dwc,
                dwc_kernel_size=dwc_kernel_size,
            )

    def _build_bias(
        self,
        B: int,
        agent_shape: Tuple[int, int],
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.bias is None:
            return None, None
        if self.bias_type == 'interp':
            return self.bias(B=B, Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk, device=device)
        # crpb
        return self.bias(B=B, agent_shape=agent_shape, Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk, device=device)

    def _forward_attention(self, q: torch.Tensor, kv: torch.Tensor, Hq, Wq, Hk, Wk, weight_map) -> torch.Tensor:
        B, Nq, C = q.shape

        q_norm = self.norm_q1(q)
        kv_norm = self.norm_kv1(kv)

        bias_a2k, bias_q2a = self._build_bias(
            B=B, agent_shape=self.attn_fwd.agent_generator.agent_shape,
            Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk, device=q.device,
        )

        out1 = q + self.drop_path(
            self.attn_fwd(
                x_q=q_norm, x_kv=kv_norm, Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk,
                bias_a2k=bias_a2k, bias_q2a=bias_q2a, weight_map=weight_map,
            ),
        )
        out1 = out1 + self.drop_path(self.mlp1(self.norm_mlp1(out1), Hq, Wq))
        return out1

    def _feedback_attention(
        self, fwd_out: torch.Tensor, q: torch.Tensor, kv: torch.Tensor, Hq, Wq, Hk, Wk, weight_map,
    ) -> torch.Tensor:
        B, Nq, C = q.shape

        q_norm = self.norm_q2(q)
        kv_norm = self.norm_kv2(kv)

        # if self.share_agent_tokens:
        #     agent2 = agent1
        # else:

        # feedback uses encoder tokens as KV (same resolution as query)
        bias_a2k2, bias_q2a2 = self._build_bias(
            B=B, agent_shape=self.attn_bwd.agent_generator.agent_shape,
            Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk, device=q.device,
        )

        out2 = fwd_out + self.drop_path(
            self.attn_bwd(
                x_q=q_norm, x_kv=kv_norm, Hq=Hq, Wq=Wq, Hk=Hk, Wk=Wk,
                bias_a2k=bias_a2k2, bias_q2a=bias_q2a2, weight_map=weight_map,
            ),
        )
        if self.use_ddpu:
            out2 = out2 + self.drop_path(self.ddpu(self.norm_ddpu(out2), Hq, Wq))
        out2 = out2 + self.drop_path(self.mlp2(self.norm_mlp2(out2), Hq, Wq))
        return out2

    def forward(
        self,
        enc_map: torch.Tensor,
        feat_maps: Sequence[torch.Tensor],
        feat_maps2: Sequence[torch.Tensor],
        replace_index: int,
        Hq: int,
        Wq: int,
        Hk: int,
        Wk: int,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              query tokens from encoder stage (B, Nq, C)
            kv:             mixed kv tokens (B, Nk, C_kv)
            enc:            encoder stage tokens used as KV in feedback pass (B, Nq, C)
            Hq,Wq:          query size
            Hk,Wk:          kv size (often deepest resolution)
            weight_map:     (B,1,Hq,Wq), boundary-guided weights for agent pooling
        Returns:
            propagated stage output
        """
        q1_token = nchw_to_nlc(enc_map)
        kv1_token = nchw_to_nlc(torch.cat(list(feat_maps), dim=1))
        fwd_out = self._forward_attention(q1_token, kv1_token, Hq, Wq, Hk, Wk, weight_map)
        if not self.use_bidirectional:
            return fwd_out

        fwd_out_map = nlc_to_nchw(fwd_out, (Hq, Wq))
        q2_token = nchw_to_nlc(torch.cat([enc_map, fwd_out_map], dim=1))
        kv2_token = nchw_to_nlc(torch.cat(list(feat_maps2), dim=1))
        fb_out = self._feedback_attention(fwd_out, q2_token, kv2_token, Hq, Wq, Hk, Wk, weight_map)

        return fwd_out + fb_out


@MODELS.register_module()
class BiAgentHead(BaseDecodeHead):
    """
    Bi-directional Agent Head (proposed).
    - 4 decoding stages (s4->s1), each stage produces 2 feature maps (out1/out2).
    - CatKey matches U-MixFormer/APFormerHead2 style: KV are concatenated pooled multi-scale features.
    - Optional fusion: use only stage_out (4 maps) or concatenate all 8 maps.

    Important: stage-wise list parameters are in order [s4, s3, s2, s1] (deep -> shallow).
    """

    def __init__(
        self,
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        agent_shapes: Sequence[Tuple[int, int]] = (7, 7, 7, 7),
        agent_token_type: str = 'avgpool',  # {'avgpool','edge_pool','learnable','hybrid'}
        bias_type: str = 'crpb',  # {'interp','crpb','none'}
        crpb_hidden_dim: int = 128,
        mlp_ratios: Sequence[Optional[float, Tuple[float, float]]] = (4.0, 4.0, 4.0, 4.0),
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        qkv_bias: bool = True,
        use_bidirectional: Sequence[bool] = (True, True, True, True),
        share_agent_tokens: bool = False,
        feedback_q_concat: bool = True,
        use_ddpu: bool = True,
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
        use_boundary_prior: bool = False,
        boundary_mid_channels: int = 32,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        # in_channels are in order [c1,c2,c3,c4] (shallow -> deep) in mmseg
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = c1_in + c2_in + c3_in + c4_in
        pool4, pool3, pool2, pool1 = pool_ratio

        # Stage order: s4->s1 means [c4,c3,c2,c1]
        self.stage_dims = [c4_in, c3_in, c2_in, c1_in]
        self.num_heads = [int(x) for x in num_heads]
        self.agent_shapes = [to_2tuple(x) for x in agent_shapes]
        self.mlp_ratios = [to_2tuple(x) for x in mlp_ratios]
        self.use_bidirectional = to_4tuple(use_bidirectional)

        assert len(self.num_heads) == 4
        assert len(self.agent_shapes) == 4
        assert len(self.mlp_ratios) == 4

        self.use_boundary_prior = bool(use_boundary_prior) and (agent_token_type in ('edge_pool', 'hybrid'))

        # Boundary prior from c1 (computed once)
        if self.use_boundary_prior:
            self.boundary_prior = BoundaryPrior(
                in_channels=c1_in,
                mid_channels=int(boundary_mid_channels),
                norm_cfg=self.norm_cfg,
            )
        else:
            self.boundary_prior = None

        self.enc1to4_downsample = Downsample(pool1 // pool4, c1_in)
        self.enc2to4_downsample = Downsample(pool2 // pool4, c2_in)
        self.enc3to4_downsample = Downsample(pool3 // pool4, c3_in)
        self.dec3to4_downsample = Downsample(pool3 // pool4, c3_in)
        self.dec2to4_downsample = Downsample(pool2 // pool4, c2_in)

        # Stage blocks
        self.blocks = ModuleList(
            [
                BiAgentBlock(
                    enc_dim=self.stage_dims[0], feats_dim=tot_channels, num_heads=self.num_heads[0],
                    agent_shape=self.agent_shapes[0], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[0], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=self.use_bidirectional[0],
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
                BiAgentBlock(
                    enc_dim=self.stage_dims[1], feats_dim=tot_channels, num_heads=self.num_heads[1],
                    agent_shape=self.agent_shapes[1], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[1], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=self.use_bidirectional[1],
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
                BiAgentBlock(
                    enc_dim=self.stage_dims[2], feats_dim=tot_channels, num_heads=self.num_heads[2],
                    agent_shape=self.agent_shapes[2], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[2], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=self.use_bidirectional[2],
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
                BiAgentBlock(
                    enc_dim=self.stage_dims[3], feats_dim=tot_channels, num_heads=self.num_heads[3],
                    agent_shape=self.agent_shapes[3], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[3], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=self.use_bidirectional[3],
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
            ],
        )

        # Final fusion and prediction
        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=int(self.channels),
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # [c1,c2,c3,c4]
        c1, c2, c3, c4 = x
        _, _, h1, w1 = c1.shape
        _, _, h2, w2 = c2.shape
        _, _, h3, w3 = c3.shape
        _, _, h4, w4 = c4.shape

        enc1to4_map = self.enc1to4_downsample(c1)
        enc2to4_map = self.enc2to4_downsample(c2)
        enc3to4_map = self.enc3to4_downsample(c3)

        # Boundary prior map from c1 (optional)
        if self.boundary_prior is not None:
            w_map1 = self.boundary_prior(c1)  # (B,1,h1,w1)
            w_map4 = resize(w_map1, size=(h4, w4), mode='bilinear', align_corners=self.align_corners)
            w_map3 = resize(w_map1, size=(h3, w3), mode='bilinear', align_corners=self.align_corners)
            w_map2 = resize(w_map1, size=(h2, w2), mode='bilinear', align_corners=self.align_corners)
        else:
            w_map1 = None
            w_map4 = None
            w_map3 = None
            w_map2 = None

        # Stage s4 (c4)
        d4_token = self.blocks[0](
            c4, (c4, enc3to4_map, enc2to4_map, enc1to4_map), (c4, enc3to4_map, enc2to4_map, enc1to4_map), 0,
            Hq=h4, Wq=w4, Hk=h4, Wk=w4, weight_map=w_map4,
        )
        d4_map = nlc_to_nchw(d4_token, (h4, w4))

        # Stage s3 (c3)  (replace c4 by stage4 output in kv)
        d3_token = self.blocks[1](
            c3, (d4_map, enc3to4_map, enc2to4_map, enc1to4_map), (c4, enc3to4_map, enc2to4_map, enc1to4_map), 1,
            Hq=h3, Wq=w3, Hk=h4, Wk=w4, weight_map=w_map3,
        )
        d3_map = nlc_to_nchw(d3_token, (h3, w3))

        # Stage s2 (c2)  (replace c4,c3 by stage4,stage3 in kv)
        dec3to4_map = self.dec3to4_downsample(d3_map)
        d2_token = self.blocks[2](
            c2, (d4_map, dec3to4_map, enc2to4_map, enc1to4_map), (c4, enc3to4_map, enc2to4_map, enc1to4_map), 2,
            Hq=h2, Wq=w2, Hk=h4, Wk=w4, weight_map=w_map2,
        )
        d2_map = nlc_to_nchw(d2_token, (h2, w2))

        # Stage s1 (c1)
        # w1_map is already at (h1,w1)
        dec2to4_map = self.dec2to4_downsample(d2_map)
        d1_token = self.blocks[3](
            c1, (d4_map, dec3to4_map, dec2to4_map, enc1to4_map), (c4, enc3to4_map, enc2to4_map, enc1to4_map), 3,
            Hq=h1, Wq=w1, Hk=h4, Wk=w4, weight_map=w_map1,
        )
        d1_map = nlc_to_nchw(d1_token, (h1, w1))

        # Final fusion
        # Upsample to c1 resolution
        d4_up = resize(d4_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        d3_up = resize(d3_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        d2_up = resize(d2_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        feats = torch.cat([d4_up, d3_up, d2_up, d1_map], dim=1)

        feats = self.linear_fuse(feats)
        seg_logits = self.cls_seg(feats)
        return seg_logits


@MODELS.register_module()
class BiAgentHeadForMSCAN(BaseDecodeHead):
    """
    Bi-directional Agent Head (proposed).
    - 4 decoding stages (s4->s1), each stage produces 2 feature maps (out1/out2).
    - CatKey matches U-MixFormer/APFormerHead2 style: KV are concatenated pooled multi-scale features.
    - Optional fusion: use only stage_out (4 maps) or concatenate all 8 maps.

    Important: stage-wise list parameters are in order [s4, s3, s2, s1] (deep -> shallow).
    """

    def __init__(
        self,
        num_heads: Sequence[int] = (8, 5, 2),
        pool_ratio: Sequence[int] = (1, 2, 4),
        agent_shapes: Sequence[Tuple[int, int]] = (7, 7, 7),
        agent_token_type: str = 'avgpool',  # {'avgpool','edge_pool','learnable','hybrid'}
        bias_type: str = 'crpb',  # {'interp','crpb','none'}
        crpb_hidden_dim: int = 128,
        mlp_ratios: Sequence[Optional[float, Tuple[float, float]]] = (4.0, 4.0, 4.0),
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        qkv_bias: bool = True,
        use_bidirectional: bool = True,
        share_agent_tokens: bool = False,
        feedback_q_concat: bool = True,
        use_ddpu: bool = True,
        use_dwc: bool = True,
        dwc_kernel_size: int = 3,
        use_boundary_prior: bool = False,
        boundary_mid_channels: int = 32,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        # in_channels are in order [c1,c2,c3,c4] (shallow -> deep) in mmseg
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = c2_in + c3_in + c4_in
        pool4, pool3, pool2 = pool_ratio

        # Stage order: s4->s1 means [c4,c3,c2,c1]
        self.stage_dims = [c4_in, c3_in, c2_in]
        self.num_heads = [int(x) for x in num_heads]
        self.agent_shapes = [to_2tuple(x) for x in agent_shapes]
        self.mlp_ratios = [to_2tuple(x) for x in mlp_ratios]

        self.use_boundary_prior = bool(use_boundary_prior) and (agent_token_type in ('edge_pool', 'hybrid'))

        # Boundary prior from c1 (computed once)
        if self.use_boundary_prior:
            self.boundary_prior = BoundaryPrior(
                in_channels=c1_in,
                mid_channels=int(boundary_mid_channels),
                norm_cfg=self.norm_cfg,
            )
        else:
            self.boundary_prior = None

        self.enc2to4_downsample = Downsample(pool2 // pool4, c2_in)
        self.enc3to4_downsample = Downsample(pool3 // pool4, c3_in)
        self.dec3to4_downsample = Downsample(pool3 // pool4, c3_in)

        # Stage blocks
        self.blocks = ModuleList(
            [
                BiAgentBlock(
                    enc_dim=self.stage_dims[0], feats_dim=tot_channels, num_heads=self.num_heads[0],
                    agent_shape=self.agent_shapes[0], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[0], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=use_bidirectional,
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
                BiAgentBlock(
                    enc_dim=self.stage_dims[1], feats_dim=tot_channels, num_heads=self.num_heads[1],
                    agent_shape=self.agent_shapes[1], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[1], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=use_bidirectional,
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
                BiAgentBlock(
                    enc_dim=self.stage_dims[2], feats_dim=tot_channels, num_heads=self.num_heads[2],
                    agent_shape=self.agent_shapes[2], agent_token_type=agent_token_type,
                    bias_type=bias_type, crpb_hidden_dim=crpb_hidden_dim,
                    mlp_ratio=self.mlp_ratios[2], drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                    qkv_bias=qkv_bias, use_bidirectional=use_bidirectional,
                    share_agent_tokens=share_agent_tokens, feedback_q_concat=feedback_q_concat,
                    use_ddpu=use_ddpu, use_dwc=use_dwc, dwc_kernel_size=dwc_kernel_size,
                ),
            ],
        )

        # Final fusion and prediction
        self.linear_fuse = ConvModule(
            in_channels=c1_in + c2_in + c3_in + c4_in,
            out_channels=int(self.channels),
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # [c1,c2,c3,c4]
        c1, c2, c3, c4 = x
        _, _, h1, w1 = c1.shape
        _, _, h2, w2 = c2.shape
        _, _, h3, w3 = c3.shape
        _, _, h4, w4 = c4.shape

        enc2to4_map = self.enc2to4_downsample(c2)
        enc3to4_map = self.enc3to4_downsample(c3)

        # Boundary prior map from c1 (optional)
        if self.boundary_prior is not None:
            w_map1 = self.boundary_prior(c1)  # (B,1,h1,w1)
            w_map4 = resize(w_map1, size=(h4, w4), mode='bilinear', align_corners=self.align_corners)
            w_map3 = resize(w_map1, size=(h3, w3), mode='bilinear', align_corners=self.align_corners)
            w_map2 = resize(w_map1, size=(h2, w2), mode='bilinear', align_corners=self.align_corners)
        else:
            w_map4 = None
            w_map3 = None
            w_map2 = None

        # Stage s4 (c4)
        d4_token = self.blocks[0](
            c4, (c4, enc3to4_map, enc2to4_map), (c4, enc3to4_map, enc2to4_map), 0,
            Hq=h4, Wq=w4, Hk=h4, Wk=w4, weight_map=w_map4,
        )
        d4_map = nlc_to_nchw(d4_token, (h4, w4))

        # Stage s3 (c3)  (replace c4 by stage4 output in kv)
        d3_token = self.blocks[1](
            c3, (d4_map, enc3to4_map, enc2to4_map), (c4, enc3to4_map, enc2to4_map), 1,
            Hq=h3, Wq=w3, Hk=h4, Wk=w4, weight_map=w_map3,
        )
        d3_map = nlc_to_nchw(d3_token, (h3, w3))

        # Stage s2 (c2)  (replace c4,c3 by stage4,stage3 in kv)
        dec3to4_map = self.dec3to4_downsample(d3_map)
        d2_token = self.blocks[2](
            c2, (d4_map, dec3to4_map, enc2to4_map), (c4, enc3to4_map, enc2to4_map), 2,
            Hq=h2, Wq=w2, Hk=h4, Wk=w4, weight_map=w_map2,
        )
        d2_map = nlc_to_nchw(d2_token, (h2, w2))

        # Final fusion
        # Upsample to c1 resolution
        d4_up = resize(d4_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        d3_up = resize(d3_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        d2_up = resize(d2_map, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        feats = torch.cat([d4_up, d3_up, d2_up, c1], dim=1)

        feats = self.linear_fuse(feats)
        seg_logits = self.cls_seg(feats)
        return seg_logits
