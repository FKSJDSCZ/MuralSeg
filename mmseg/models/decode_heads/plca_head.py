# ---------------------------------------------------------------
# Copyright (c) 2021, Nota AI GmbH. All rights reserved.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize, nlc_to_nchw, nchw_to_nlc
import math


class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModule):
    def __init__(self, pool_ratio=[1, 2, 4, 8], dim=[256, 160, 64, 32]):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = ModuleList(
            [nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if
             self.pool_ratio[i] > 1],
        )
        self.pool_list = ModuleList(
            [nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio))
             if self.pool_ratio[i] > 1],
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

    def forward(self, x):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        return torch.cat(out_list, dim=1)


class CatKeyMulti(BaseModule):
    def __init__(self, pool_ratio=[1, 2, 4, 8], dim=[256, 160, 64, 32], num_feat=4):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = ModuleList(
            [nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(len(self.pool_ratio)) if
             self.pool_ratio[i] > 1],
        )
        self.pool_list = ModuleList(
            [nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(len(self.pool_ratio))
             if self.pool_ratio[i] > 1],
        )
        for _ in range(num_feat):
            self.sr_list.append(nn.Conv2d(dim[1], dim[1], kernel_size=1, stride=1))
            self.pool_list.append(nn.AvgPool2d(self.pool_ratio[1], self.pool_ratio[1], ceil_mode=True))

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

    def forward(self, x, xMulti):
        out_list = []
        cnt = 0
        for i in range(len(self.pool_ratio)):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](x[i])))
                cnt += 1
            else:
                out_list.append(x[i])
        for l in range(len(xMulti)):  # for the middle feature at stage 3
            out_list.append(self.sr_list[cnt](self.pool_list[cnt](xMulti[l])))
            cnt += 1
        return torch.cat(out_list, dim=1)


class CrossAttention(BaseModule):
    def __init__(
        self, dim1, dim2, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16,
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

    def forward(self, x, y, H2, W2):
        B1, N1, C1 = x.shape
        B2, N2, C2 = y.shape
        q = self.q(x).reshape(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3)

        if self.pool_ratio >= 0:
            # x_ = y.permute(0, 2, 1).reshape(B2, C2, H2, W2)
            # x_ = self.sr(self.pool(x_)).reshape(B2, C2, -1).permute(0, 2, 1)
            x_ = self.norm(y)
            x_ = self.act(y)
        else:
            x_ = y

        kv = self.kv(x_).reshape(B1, -1, 2, self.num_heads, C1 // self.num_heads).permute(
            2, 0, 3, 1, 4,
        )  # 여기에다가 rollout을 넣는다면?
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B1, N1, C1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(BaseModule):
    def __init__(
        self, dim1, dim2, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm2 = norm_layer(dim2)
        self.norm3 = norm_layer(dim1)

        self.attn = CrossAttention(dim1=dim1, dim2=dim2, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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

    def forward(self, x, y, H2, W2, H1, W1):
        x = self.norm1(x)
        y = self.norm2(y)
        x = x + self.drop_path(self.attn(x, y, H2, W2))  # self.norm2(y)이 F1에 대한 값
        x = self.norm3(x)
        x = x + self.drop_path(self.mlp(x, H1, W1))

        # x = x + self.drop_path(self.attn(self.norm1(x), self.norm2(y), H2, W2)) #self.norm2(y)이 F1에 대한 값
        # x = x + self.drop_path(self.mlp(self.norm3(x), H1, W1))

        return x


class CatKey_single(BaseModule):
    def __init__(self, pool_ratio=1, dim=1):
        super().__init__()
        self.pool_ratio = pool_ratio
        self.sr_list = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.pool_list = nn.AvgPool2d(self.pool_ratio, self.pool_ratio, ceil_mode=True)

    def forward(self, x):
        return self.sr_list(self.pool_list(x))


# ================================================================
# Prior-guided Low-rank Cross-Attention (PLCA) for U-MixFormer decoder
# - Low-rank Q/K interaction (generalizes SCASeg strip cross-attn, r=1)
# - Coordinate prior bias for coarse-to-fine alignment
# - Optional edge(high-pass) similarity branch (freq-inspired, no gating)
# - Optional pooled-agent factorized attention (agent-inspired, lightweight)
# All ablation knobs are exposed in the decode head init (no nested dicts).
# ================================================================


def _to_stage_list(x, name='arg'):
    """Convert a scalar or tuple/list to a 4-stage list in order [s4,s3,s2,s1]."""
    if isinstance(x, (list, tuple)):
        assert len(x) == 4, f'{name} must have length 4 in order [s4,s3,s2,s1], got {len(x)}'
        return list(x)
    return [x, x, x, x]


def _make_coord_grid(H, W, device, dtype):
    """Token center coords in [-1,1], shape [N,2] with token order matching flatten(2).transpose."""
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H * 2.0 - 1.0
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W * 2.0 - 1.0
    # torch>=1.10 supports indexing argument; but keep backward compatible.
    try:
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    except TypeError:  # pragma: no cover
        yy, xx = torch.meshgrid(ys, xs)
    coords = torch.stack([yy, xx], dim=-1).view(-1, 2)  # [H*W,2] with W fastest
    return coords


def _high_pass_avg(x_nchw, k=3):
    """Parameter-free high-pass filter: x - avgpool(x)."""
    if k <= 1:
        return x_nchw
    return x_nchw - F.avg_pool2d(x_nchw, kernel_size=k, stride=1, padding=k // 2)


class DirectionalDilatedPU(BaseModule):
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


class PLCrossAttention(BaseModule):
    """Prior-guided Low-rank Cross-Attention.

    Args:
        dim_q (int): channels of query tokens.
        dim_kv (int): channels of kv tokens.
        num_heads (int): attention heads.
        qk_embed_dim (int): low-rank dim r for Q/K interaction per head (r=1 -> strip).
        use_coord_bias (bool): add Gaussian distance bias between query/key coordinates.
        coord_bias_scale (float): scale for coord bias term.
        coord_bias_sigma (float): initial sigma for Gaussian (in normalized coord space).
        coord_bias_learnable (bool): whether sigma is learnable (per head).
        use_edge_bias (bool): add edge(high-pass) similarity branch to attention logits.
        edge_bias_scale (float): scale for edge similarity term.
        edge_pool_kernel (int): kernel size for high-pass filter.
        use_agent (bool): add pooled-agent factorized attention branch.
        agent_pool_size (int): adaptive pool size P, number of agent tokens = P*P (0 disables).
        agent_out_scale (float): scale for agent branch output.
    """

    def __init__(
        self,
        dim_q,
        dim_kv,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        qk_embed_dim=8,
        # coord prior
        use_coord_bias=True,
        coord_bias_scale=1.0,
        coord_bias_sigma=0.8,
        coord_bias_learnable=True,
        # agent factorization
        use_agent=False,
        agent_pool_size=0,
        agent_out_scale=1.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert dim_q % num_heads == 0, f'dim_q {dim_q} should be divisible by num_heads {num_heads}.'
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads

        # ---- Low-rank Q/K interaction dim (generalizes SCASeg r=1) ----
        self.qk_embed_dim = int(qk_embed_dim)
        assert self.qk_embed_dim > 0, 'qk_embed_dim must be positive'
        # scale for dot-product in r-dim space
        self.scale = self.qk_embed_dim ** -0.5

        self.q_content = nn.Linear(dim_q, num_heads * self.qk_embed_dim, bias=qkv_bias)
        self.k_content = nn.Linear(dim_kv, num_heads * self.qk_embed_dim, bias=qkv_bias)
        self.v = nn.Linear(dim_kv, dim_q, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        # ---- Coordinate prior bias (coarse-to-fine alignment) ----
        self.use_coord_bias = bool(use_coord_bias)
        self.coord_bias_scale = float(coord_bias_scale)
        self.coord_bias_learnable = bool(coord_bias_learnable)
        if self.use_coord_bias and self.coord_bias_scale != 0:
            # Per-head sigma is important because different heads learn different receptive ranges.
            init_sigma = float(coord_bias_sigma)
            if self.coord_bias_learnable:
                # parameterize with softplus for positivity
                self.coord_log_sigma = nn.Parameter(torch.full((num_heads,), math.log(math.exp(init_sigma) - 1.0)))
            else:
                self.register_buffer('coord_sigma', torch.full((num_heads,), init_sigma), persistent=False)
            # cache for coord grids to avoid recomputation under fixed input size
            self._coord_cache = {}  # (Hq,Wq,Hk,Wk,device,dtype) -> (dist2 [Nq,Nk])
        else:
            self._coord_cache = None

        # ---- Pooled-agent factorized attention (agent-inspired) ----
        self.use_agent = bool(use_agent)
        self.agent_pool_size = int(agent_pool_size)
        self.agent_out_scale = float(agent_out_scale)
        if self.use_agent and self.agent_pool_size > 0 and self.agent_out_scale != 0:
            # 2nd-step agent query projection (agents -> qk space)
            self.q_agent = nn.Linear(dim_kv, num_heads * self.qk_embed_dim, bias=qkv_bias)
        else:
            self.q_agent = None

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

    def _get_coord_bias(self, Hq, Wq, Hk, Wk, device, dtype):
        """Return coord bias tensor with shape [heads, Nq, Nk]."""
        if (not self.use_coord_bias) or self.coord_bias_scale == 0:
            return None

        key = (Hq, Wq, Hk, Wk, device, dtype)
        if self._coord_cache is not None and key in self._coord_cache:
            dist2 = self._coord_cache[key]
        else:
            cq = _make_coord_grid(Hq, Wq, device=device, dtype=dtype)  # [Nq,2]
            ck = _make_coord_grid(Hk, Wk, device=device, dtype=dtype)  # [Nk,2]
            # [Nq,Nk,2] then squared distance
            dist2 = (cq[:, None, :] - ck[None, :, :]).pow(2).sum(-1)  # [Nq,Nk]
            if self._coord_cache is not None:
                self._coord_cache[key] = dist2

        if self.coord_bias_learnable:
            sigma = F.softplus(self.coord_log_sigma) + 1e-6  # [heads]
        else:
            sigma = self.coord_sigma  # [heads]
        inv_sigma2 = 1.0 / (sigma * sigma)  # [heads]
        bias = -dist2[None, :, :] * inv_sigma2[:, None, None]  # [heads,Nq,Nk]
        bias = bias * self.coord_bias_scale
        return bias

    def forward(self, x_q, x_kv, Hk, Wk, Hq, Wq):
        """
        x_q:  [B, Nq, Cq]
        x_kv: [B, Nk, Ckv]
        Hk,Wk: spatial size of kv tokens (Nk=Hk*Wk)
        Hq,Wq: spatial size of query tokens (Nq=Hq*Wq)
        """
        B, Nq, Cq = x_q.shape
        B2, Nk, Ckv = x_kv.shape
        assert B == B2, 'Batch size mismatch between query and kv.'
        # ---- Main low-rank cross attention ----
        q = self.q_content(x_q).view(B, Nq, self.num_heads, self.qk_embed_dim).permute(0, 2, 1, 3)  # [B,h,Nq,r]
        k = self.k_content(x_kv).view(B, Nk, self.num_heads, self.qk_embed_dim).permute(0, 2, 1, 3)  # [B,h,Nk,r]
        v = self.v(x_kv).view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,h,Nk,d]

        # Build agent tokens by adaptive avg pooling on KV feature map.
        kv_map = x_kv.transpose(1, 2).reshape(B, Ckv, Hk, Wk)
        agents_map = F.adaptive_avg_pool2d(kv_map, output_size=(self.agent_pool_size, self.agent_pool_size))
        Na = self.agent_pool_size * self.agent_pool_size
        agents = agents_map.flatten(2).transpose(1, 2)  # [B,Na,Ckv]

        # Step-1: Q -> A
        k_a = self.k_content(agents).view(B, Na, self.num_heads, self.qk_embed_dim).permute(
            0, 2, 1, 3,
        )  # [B,h,Na,r]
        attn_q2a = (q @ k_a.transpose(-2, -1)) * self.scale
        # Coordinate prior
        coord_bias_q2a = self._get_coord_bias(
            Hq, Wq, self.agent_pool_size, self.agent_pool_size, device=attn_q2a.device, dtype=attn_q2a.dtype,
        )
        if coord_bias_q2a is not None:
            attn_q2a += coord_bias_q2a[None, :, :, :]  # broadcast B
        attn_q2a = attn_q2a.softmax(dim=-1)  # [B,h,Nq,Na]
        attn_q2a = self.attn_drop(attn_q2a)

        # Step-2: A -> K (agents as query)
        q_a = self.q_agent(agents).view(B, Na, self.num_heads, self.qk_embed_dim).permute(0, 2, 1, 3)  # [B,h,Na,r]
        attn_a2k = (q_a @ k.transpose(-2, -1)) * self.scale
        # Coordinate prior
        coord_bias_a2k = self._get_coord_bias(
            self.agent_pool_size, self.agent_pool_size, Hk, Wk, device=attn_a2k.device, dtype=attn_a2k.dtype,
        )
        if coord_bias_a2k is not None:
            attn_a2k += coord_bias_a2k[None, :, :, :]  # broadcast B
        attn_a2k = attn_a2k.softmax(dim=-1)  # [B,h,Na,Nk]
        attn_a2k = self.attn_drop(attn_a2k)

        mem = attn_a2k @ v  # [B,h,Na,d]
        out = (attn_q2a @ mem).transpose(1, 2).reshape(B, Nq, Cq)  # [B,Nq,Cq]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PLBlock(BaseModule):
    """Transformer-like block with PLCA attention + MixFFN (DWConv MLP)."""

    def __init__(
        self,
        dim1,
        dim2,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        # PLCA knobs
        qk_embed_dim=8,
        use_coord_bias=True,
        coord_bias_scale=1.0,
        coord_bias_sigma=0.8,
        coord_bias_learnable=True,
        use_agent=False,
        agent_pool_size=0,
        agent_out_scale=1.0,
        use_ddpu=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.norm_q = norm_layer(dim1)
        self.norm_kv = norm_layer(dim2)
        self.norm_ffn = norm_layer(dim1)

        self.attn = PLCrossAttention(
            dim_q=dim1,
            dim_kv=dim2,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_embed_dim=qk_embed_dim,
            use_coord_bias=use_coord_bias,
            coord_bias_scale=coord_bias_scale,
            coord_bias_sigma=coord_bias_sigma,
            coord_bias_learnable=coord_bias_learnable,
            use_agent=use_agent,
            agent_pool_size=agent_pool_size,
            agent_out_scale=agent_out_scale,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        # Reuse the original MixFFN (Mlp + DWConv) defined above in this file.
        self.ffn = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # ---- Directional Dilated Perception Unit(DDPU) ----
        self.use_ddpu = use_ddpu
        if self.use_ddpu:
            self.norm_ddpu = norm_layer(dim1)
            self.ddpu = DirectionalDilatedPU(dim=dim1)
        else:
            self.norm_ddpu = None
            self.ddpu = None

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

    def forward(self, x, kv, Hk, Wk, Hq, Wq):
        x = x + self.drop_path(self.attn(self.norm_q(x), self.norm_kv(kv), Hk, Wk, Hq, Wq))
        if self.use_ddpu:
            x = x + self.drop_path(self.ddpu(self.norm_ddpu(x), Hq, Wq))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x), Hq, Wq))
        return x


@MODELS.register_module()
class PLCAHead(BaseDecodeHead):
    """APFormerHead2 with ablation-ready PLCA blocks.

    Notes:
        - Stage order for list params is [s4, s3, s2, s1] (deep->shallow).
        - No nested dict for ablation knobs; every knob is an init argument.
        - Unused modules are not instantiated (edge/agent/coord are conditional).
    """

    def __init__(
        self,
        # ---- original decoder knobs (no nested dict) ----
        num_heads=(8, 5, 2, 1),  # [s4,s3,s2,s1]
        pool_ratio=(1, 2, 4, 8),  # for CatKey, aligns [c4,c3,c2,c1] -> c4
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.1,
        # ---- ablation: choose attention impl ----
        attn_impl='plca',  # {'plca','vanilla'}
        # ---- PLCA knobs (stage-wise) ----
        qk_embed_dims=(8, 8, 4, 4),  # r per head, [s4,s3,s2,s1]
        use_coord_bias=True,
        coord_bias_scales=(1.0, 1.0, 1.0, 1.0),
        coord_bias_sigmas=(0.9, 0.7, 0.5, 0.4),
        coord_bias_learnable=True,
        use_agent=False,
        agent_pool_sizes=(0, 0, 0, 0),  # P per stage, tokens=P*P (0 disables)
        agent_out_scales=(1.0, 1.0, 1.0, 1.0),
        use_ddpu=False,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = sum(self.in_channels)

        # normalize stage-wise lists (order s4->s1)
        num_heads = _to_stage_list(num_heads, 'num_heads')
        qk_embed_dims = _to_stage_list(qk_embed_dims, 'qk_embed_dims')
        coord_bias_scales = _to_stage_list(coord_bias_scales, 'coord_bias_scales')
        coord_bias_sigmas = _to_stage_list(coord_bias_sigmas, 'coord_bias_sigmas')
        agent_pool_sizes = _to_stage_list(agent_pool_sizes, 'agent_pool_sizes')
        agent_out_scales = _to_stage_list(agent_out_scales, 'agent_out_scales')

        self.attn_impl = attn_impl.lower()

        # ---- build attention blocks (ablation-ready) ----
        if self.attn_impl == 'vanilla':
            # Use the official implementation blocks defined above.
            self.attn_c4 = Block(
                dim1=c4_in, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, pool_ratio=8,
            )
            self.attn_c3 = Block(
                dim1=c3_in, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, pool_ratio=4,
            )
            self.attn_c2 = Block(
                dim1=c2_in, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, pool_ratio=2,
            )
            self.attn_c1 = Block(
                dim1=c1_in, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, pool_ratio=1,
            )
        elif self.attn_impl == 'plca':
            self.attn_c4 = PLBlock(
                dim1=c4_in, dim2=tot_channels, num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                qk_embed_dim=qk_embed_dims[0],
                use_coord_bias=use_coord_bias, coord_bias_scale=coord_bias_scales[0],
                coord_bias_sigma=coord_bias_sigmas[0], coord_bias_learnable=coord_bias_learnable,
                use_agent=use_agent, agent_pool_size=agent_pool_sizes[0],
                agent_out_scale=agent_out_scales[0],
                use_ddpu=use_ddpu,
            )
            self.attn_c3 = PLBlock(
                dim1=c3_in, dim2=tot_channels, num_heads=num_heads[1], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                qk_embed_dim=qk_embed_dims[1],
                use_coord_bias=use_coord_bias, coord_bias_scale=coord_bias_scales[1],
                coord_bias_sigma=coord_bias_sigmas[1], coord_bias_learnable=coord_bias_learnable,
                use_agent=use_agent, agent_pool_size=agent_pool_sizes[1],
                agent_out_scale=agent_out_scales[1],
                use_ddpu=use_ddpu,
            )
            self.attn_c2 = PLBlock(
                dim1=c2_in, dim2=tot_channels, num_heads=num_heads[2], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                qk_embed_dim=qk_embed_dims[2],
                use_coord_bias=use_coord_bias, coord_bias_scale=coord_bias_scales[2],
                coord_bias_sigma=coord_bias_sigmas[2], coord_bias_learnable=coord_bias_learnable,
                use_agent=use_agent, agent_pool_size=agent_pool_sizes[2],
                agent_out_scale=agent_out_scales[2],
                use_ddpu=use_ddpu,
            )
            self.attn_c1 = PLBlock(
                dim1=c1_in, dim2=tot_channels, num_heads=num_heads[3], mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                qk_embed_dim=qk_embed_dims[3],
                use_coord_bias=use_coord_bias, coord_bias_scale=coord_bias_scales[3],
                coord_bias_sigma=coord_bias_sigmas[3], coord_bias_learnable=coord_bias_learnable,
                use_agent=use_agent, agent_pool_size=agent_pool_sizes[3],
                agent_out_scale=agent_out_scales[3],
                use_ddpu=use_ddpu,
            )
        else:
            raise ValueError(f'Unknown attn_impl={attn_impl}, supported: ["plca","vanilla"]')

        # ---- Mixed key/value construction ----
        self.cat_key1 = CatKey(pool_ratio=list(pool_ratio), dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key2 = CatKey(pool_ratio=list(pool_ratio), dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key3 = CatKey(pool_ratio=list(pool_ratio), dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key4 = CatKey(pool_ratio=list(pool_ratio), dim=[c4_in, c3_in, c2_in, c1_in])

        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, [1/4,1/8,1/16,1/32]
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # ---- Stage s4 ----
        c_key = self.cat_key1([c4, c3, c2, c1])
        c_key_tok = c_key.flatten(2).transpose(1, 2)  # [B, H4*W4, Csum]
        c4_tok = c4.flatten(2).transpose(1, 2)  # [B, H4*W4, C4]
        _c4 = self.attn_c4(c4_tok, c_key_tok, h4, w4, h4, w4)

        # ---- Stage s3 ----
        _c4_map = _c4.permute(0, 2, 1).reshape(n, -1, h4, w4)
        c_key_tok = self.cat_key2([_c4_map, c3, c2, c1]).flatten(2).transpose(1, 2)
        c3_tok = c3.flatten(2).transpose(1, 2)
        _c3 = self.attn_c3(c3_tok, c_key_tok, h4, w4, h3, w3)

        # ---- Stage s2 ----
        _c3_map = _c3.permute(0, 2, 1).reshape(n, -1, h3, w3)
        c_key_tok = self.cat_key3([_c4_map, _c3_map, c2, c1]).flatten(2).transpose(1, 2)
        c2_tok = c2.flatten(2).transpose(1, 2)
        _c2 = self.attn_c2(c2_tok, c_key_tok, h4, w4, h2, w2)

        # ---- Stage s1 ----
        _c2_map = _c2.permute(0, 2, 1).reshape(n, -1, h2, w2)
        c_key_tok = self.cat_key4([_c4_map, _c3_map, _c2_map, c1]).flatten(2).transpose(1, 2)
        c1_tok = c1.flatten(2).transpose(1, 2)
        _c1 = self.attn_c1(c1_tok, c_key_tok, h4, w4, h1, w1)

        # ---- Fuse ----
        _c4_up = resize(_c4_map, size=(h1, w1), mode='bilinear', align_corners=False)
        _c3_up = resize(_c3_map, size=(h1, w1), mode='bilinear', align_corners=False)
        _c2_up = resize(_c2_map, size=(h1, w1), mode='bilinear', align_corners=False)
        _c1_map = _c1.permute(0, 2, 1).reshape(n, -1, h1, w1)

        fused = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1_map], dim=1))
        out = self.cls_seg(fused)
        return out
