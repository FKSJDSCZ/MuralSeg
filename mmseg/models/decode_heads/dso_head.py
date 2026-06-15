# This file implements a novel U-MixFormer-style decoder head with an improved
# attention interaction module: Dual-Source Ortho Boundary-Augmented Mix Attention (DSO-BAMA).
#
# Design goals:
#   - Higher mIoU on ADE20K / Cityscapes etc (accuracy-first).
#   - Keep decoder efficient by using low-res mixed KV + a small set of informative agent tokens.
#   - Avoid hard gating / weighted fusion between branches (user observation #2).
#   - Avoid naive channel-splitting mixture (user observation #1).
#
# Key ideas:
#   (1) Dual-source agents: agents are pooled from BOTH query and KV (motivated by user obs #3).
#   (2) Ortho agents: per-head agent tokens are orthonormalized (Cholesky whitening) to reduce redundancy.
#   (3) Boundary agents: a small set of boundary-focused tokens distilled from high-res shallow feature (C1),
#       injected into the same attention softmax as additional KV tokens (no gating).
#
# The implementation follows mmsegmentation coding conventions and reuses BaseDecodeHead, BaseModuleInit,
# ModuleList, DropPath, ConvModule, resize, nlc_to_nchw/nchw_to_nlc when available.
#
# NOTE: Place this file under `mmseg/models/decode_heads/` (or a custom plugin package) to enable
# relative import `from ..utils import resize`.

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule, ModuleList
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
    """Depthwise conv used inside Mix-FFN to inject locality / position."""

    def __init__(self, dim: int, kernel_size: int = 3, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, groups=dim, bias=True,
        )

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        # x: [B, N, C]
        h, w = hw_shape
        x = nlc_to_nchw(x, (h, w))
        x = self.dwconv(x)
        x = nchw_to_nlc(x)
        return x


class MixMlp(BaseModuleInit):
    """MLP with depthwise conv (Mix-FFN style)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        dwconv_kernel_size: int = 3,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, kernel_size=dwconv_kernel_size)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, hw_shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModuleInit):
    """Concatenate multi-stage features as KV after aligning spatial sizes by avg-pooling.

    The pool ratios are defined w.r.t the deepest stage (s4). For example, if input feature
    resolutions are {1/32,1/16,1/8,1/4} and s4 is 1/32, then pool_ratio should be (1,2,4,8)
    to downsample s3/s2/s1 to s4 resolution before concatenation.
    """

    def __init__(
        self,
        pool_ratio: Sequence[int],
        dim: Sequence[int],
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(pool_ratio) == len(dim)
        self.pool_ratio = list(pool_ratio)

        self.pool_list = ModuleList()
        self.sr_list = ModuleList()
        for r, c in zip(self.pool_ratio, dim):
            if r > 1:
                self.pool_list.append(nn.AvgPool2d(r, r, ceil_mode=True))
                self.sr_list.append(nn.Conv2d(c, c, kernel_size=1, stride=1, bias=True))

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == len(self.pool_ratio)
        outs = []
        cnt = 0
        for i, r in enumerate(self.pool_ratio):
            if r > 1:
                outs.append(self.sr_list[cnt](self.pool_list[cnt](feats[i])))
                cnt += 1
            else:
                outs.append(feats[i])
        return torch.cat(outs, dim=1)


class EdgeAwareTokenPool(BaseModuleInit):
    """Edge-aware pooling to distill a small set of boundary tokens from high-res feature map.

    We avoid an extra learnable branch; instead we compute a lightweight, differentiable
    edge weight map via a high-pass residual (x - avgpool(x)), and do *weighted* adaptive
    pooling:
        B = AAP(x * e) / (AAP(e) + eps)

    Output tokens are flattened as [B, Nb, C_in], where Nb = pool_h * pool_w.

    Args:
        pool_size: (hb, wb) output grid size; if None, this module is not used.
        lowpass_kernel: kernel size for low-pass avgpool; default 3.
        eps: numerical stabilizer.
        use_abs: whether to use abs(highpass) to compute edge magnitude.
    """

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = (7, 7),
        lowpass_kernel: int = 3,
        eps: float = 1e-6,
        use_abs: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.pool_size = to_2tuple(pool_size)
        self.lowpass_kernel = int(lowpass_kernel)
        self.eps = float(eps)
        self.use_abs = bool(use_abs)
        self.pool = nn.AdaptiveAvgPool2d(self.pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        k = self.lowpass_kernel
        # low-pass, keep resolution
        lp = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        hp = x - lp
        if self.use_abs:
            e = torch.mean(torch.abs(hp), dim=1, keepdim=True)  # [B,1,H,W]
        else:
            e = torch.mean(hp * hp, dim=1, keepdim=True)

        num = self.pool(x * e)
        den = self.pool(e)
        pooled = num / (den + self.eps)  # [B,C,hb,wb]
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()  # [B,Nb,C]
        return tokens


def _orthonormalize_tokens(
    a: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Orthonormalize tokens along the token dimension (rows).

    Args:
        a: [B, H, N, D] (per head tokens)
        eps: diagonal jitter for numerical stability.

    Returns:
        a_ortho: [B, H, N, D], such that a_ortho @ a_ortho^T ~= I (row-orthonormal)
    """
    b, h, n, d = a.shape
    if n == 1:
        return a

    a_fp32 = a.float()
    # Gram matrix: [B,H,N,N]
    g = a_fp32 @ a_fp32.transpose(-2, -1)
    # add eps*I
    eye = torch.eye(n, device=a.device, dtype=a_fp32.dtype).view(1, 1, n, n)
    g = g + eps * eye
    # Cholesky (lower)
    l = torch.linalg.cholesky(g)
    # Solve L * x = a  => x = L^{-1} a  (row-orthonormalization)
    a_ortho = torch.linalg.solve_triangular(l, a_fp32, upper=False)
    return a_ortho.to(dtype=a.dtype)


class DualSourceOrthoBoundaryAugCrossAgentAttention(BaseModuleInit):
    """Dual-source Ortho Boundary-Augmented Cross-Agent Attention (DSO-BAMA).

    It extends the "AugmentedCrossAgentAttention" idea (KV-augmentation) by:
      1) Dual-source agents: agent tokens are pooled from both query and KV, then added.
      2) Ortho agents: per-head agents are orthonormalized to increase diversity.
      3) Boundary agents: boundary-focused tokens distilled from a shallow feature map are injected
         into the same softmax as extra KV tokens.

    Forward:
        Input:
            x_q: [B, Nq, Cq]
            x_kv: [B, Nk, Ckv]
            hw_q: (Hq, Wq)
            hw_kv: (Hk, Wk)  # typically Hk=Wk=H4=W4
            boundary_tokens (optional): [B, Nb, Cb]
        Output:
            out: [B, Nq, Cq]
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        kv_agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        boundary_in_channels: Optional[int] = None,
        boundary_proj: bool = True,
        use_kv_agent: bool = True,
        use_learnable_agent: bool = False,
        use_ortho_agent: bool = True,
        ortho_eps: float = 1e-4,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert dim_q % num_heads == 0, f"dim_q={dim_q} must be divisible by num_heads={num_heads}"
        self.dim_q = int(dim_q)
        self.dim_kv = int(dim_kv)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim_q // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.use_kv_agent = bool(use_kv_agent)
        self.use_learnable_agent = bool(use_learnable_agent)
        self.use_ortho_agent = bool(use_ortho_agent)
        self.ortho_eps = float(ortho_eps)

        # Projections for main attention
        self.q = nn.Linear(self.dim_q, self.dim_q, bias=qkv_bias)
        self.kv = nn.Linear(self.dim_kv, self.dim_q * 2, bias=qkv_bias)

        # Projections for dual-source agents
        self.agent_pool_size = to_2tuple(agent_pool_size)
        self.agent_pool = nn.AdaptiveAvgPool2d(self.agent_pool_size)

        self.kv_agent_pool_size = to_2tuple(kv_agent_pool_size)
        self.kv_agent_pool = nn.AdaptiveAvgPool2d(self.kv_agent_pool_size)
        if self.use_kv_agent:
            # Project kv tokens to dim_q before pooling (to match q-agent dimension)
            self.kv_to_q = nn.Linear(self.dim_kv, self.dim_q, bias=True)

        if self.use_learnable_agent:
            agent_num = self.agent_pool_size[0] * self.agent_pool_size[1]
            self.learnable_agent = nn.Parameter(torch.zeros(1, agent_num, self.dim_q))
            nn.init.trunc_normal_(self.learnable_agent, std=0.02)

        # Boundary token projection (optional)
        self.boundary_in_channels = boundary_in_channels
        self.boundary_proj_enabled = bool(boundary_proj)
        if boundary_in_channels is not None and self.boundary_proj_enabled:
            self.boundary_proj = nn.Linear(boundary_in_channels, self.dim_q, bias=True)
        else:
            self.boundary_proj = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim_q, self.dim_q, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_dwconv = bool(use_dwconv)
        if self.use_dwconv:
            self.dwconv = nn.Conv2d(
                in_channels=self.dim_q,
                out_channels=self.dim_q,
                kernel_size=dwconv_kernel_size,
                padding=dwconv_kernel_size // 2,
                groups=self.dim_q,
                bias=True,
            )

        self.softmax = nn.Softmax(dim=-1)

    def _make_q_agents(self, q_tokens: torch.Tensor, hw_q: Tuple[int, int]) -> torch.Tensor:
        # q_tokens: [B, Nq, Cq]
        b, n, c = q_tokens.shape
        h, w = hw_q
        q_map = q_tokens.transpose(1, 2).reshape(b, c, h, w)
        a = self.agent_pool(q_map).reshape(b, c, -1).transpose(1, 2).contiguous()  # [B,Na,Cq]
        if self.use_learnable_agent:
            a = a + self.learnable_agent.to(dtype=a.dtype, device=a.device)
        return a

    def _make_kv_agents(self, kv_tokens: torch.Tensor, hw_kv: Tuple[int, int]) -> torch.Tensor:
        # kv_tokens: [B, Nk, Ckv]
        b, n, c = kv_tokens.shape
        hk, wk = hw_kv
        kv_q = self.kv_to_q(kv_tokens)  # [B,Nk,Cq]
        kv_map = kv_q.transpose(1, 2).reshape(b, self.dim_q, hk, wk)
        a_kv = self.kv_agent_pool(kv_map).reshape(b, self.dim_q, -1).transpose(1, 2).contiguous()  # [B,Na,Cq]
        return a_kv

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, Cq] -> [B, H, N, D]
        b, n, c = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D] -> [B, N, Cq]
        b, h, n, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, n, h * d).contiguous()
        return x

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hw_q: Tuple[int, int],
        hw_kv: Tuple[int, int],
        boundary_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, nq, _ = x_q.shape
        _, nk, _ = x_kv.shape

        # Project Q, K, V
        q = self.q(x_q)  # [B,Nq,Cq]
        kv = self.kv(x_kv)  # [B,Nk,2*Cq]
        kv = kv.reshape(b, nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]  # [B,H,Nk,D]
        qh = self._split_heads(q)  # [B,H,Nq,D]

        # ----- Dual-source semantic agents -----
        a_q = self._make_q_agents(q, hw_q)  # [B,Na,Cq]
        if self.use_kv_agent:
            a_kv = self._make_kv_agents(x_kv, hw_kv)  # [B,Na,Cq]
            a = a_q + a_kv  # direct add (avoid hard gating)
        else:
            a = a_q

        ah = self._split_heads(a)  # [B,H,Na,D]
        if self.use_ortho_agent:
            ah = _orthonormalize_tokens(ah, eps=self.ortho_eps)

        # Agent summary values: V_a = softmax(A K^T) V
        attn_a = (ah @ k.transpose(-2, -1)) * self.scale  # [B,H,Na,Nk]
        attn_a = self.softmax(attn_a)
        attn_a = self.attn_drop(attn_a)
        v_a = attn_a @ v  # [B,H,Na,D]

        # ----- Boundary agents (optional) -----
        if boundary_tokens is not None and self.boundary_proj is not None:
            bnd = self.boundary_proj(boundary_tokens)  # [B,Nb,Cq]
            bnd_h = self._split_heads(bnd)  # [B,H,Nb,D]
            if self.use_ortho_agent:
                bnd_h = _orthonormalize_tokens(bnd_h, eps=self.ortho_eps)

            attn_b = (bnd_h @ k.transpose(-2, -1)) * self.scale  # [B,H,Nb,Nk]
            attn_b = self.softmax(attn_b)
            attn_b = self.attn_drop(attn_b)
            v_b = attn_b @ v  # [B,H,Nb,D]

            # Augment KV with {semantic agent tokens, boundary tokens}
            k_aug = torch.cat([k, ah, bnd_h], dim=2)
            v_aug = torch.cat([v, v_a, v_b], dim=2)
        else:
            k_aug = torch.cat([k, ah], dim=2)
            v_aug = torch.cat([v, v_a], dim=2)

        # Final attention: softmax(Q K_aug^T) V_aug
        attn = (qh @ k_aug.transpose(-2, -1)) * self.scale  # [B,H,Nq,Nk+...]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = attn @ v_aug  # [B,H,Nq,D]
        out = self._merge_heads(out)  # [B,Nq,Cq]

        # Local enhancement via DWConv on query map (optional)
        if self.use_dwconv:
            hq, wq = hw_q
            q_map = q.transpose(1, 2).reshape(b, self.dim_q, hq, wq)
            q_map = self.dwconv(q_map)
            q_local = q_map.flatten(2).transpose(1, 2).contiguous()
            out = out + q_local

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class UMixFormerDSOBBlock(BaseModuleInit):
    """A U-MixFormer decoder block with DSO-BAMA attention + MixMlp."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        qkv_bias: bool = False,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        kv_agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        boundary_in_channels: Optional[int] = None,
        use_boundary: bool = True,
        use_kv_agent: bool = True,
        use_learnable_agent: bool = False,
        use_ortho_agent: bool = True,
        ortho_eps: float = 1e-4,
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
        dwconv_mlp_kernel_size: int = 3,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)

        self.attn = DualSourceOrthoBoundaryAugCrossAgentAttention(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            agent_pool_size=agent_pool_size,
            kv_agent_pool_size=kv_agent_pool_size,
            boundary_in_channels=boundary_in_channels if use_boundary else None,
            boundary_proj=True,
            use_kv_agent=use_kv_agent,
            use_learnable_agent=use_learnable_agent,
            use_ortho_agent=use_ortho_agent,
            ortho_eps=ortho_eps,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_dwconv=use_dwconv,
            dwconv_kernel_size=dwconv_kernel_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_q)

        mlp_hidden = int(dim_q * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim_q,
            hidden_features=mlp_hidden,
            drop=drop,
            dwconv_kernel_size=dwconv_mlp_kernel_size,
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        hw_q: Tuple[int, int],
        hw_kv: Tuple[int, int],
        boundary_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm
        x_q = x_q + self.drop_path(self.attn(self.norm_q(x_q), self.norm_kv(x_kv), hw_q=hw_q, hw_kv=hw_kv, boundary_tokens=boundary_tokens))
        x_q = x_q + self.drop_path(self.mlp(self.norm2(x_q), hw_q))
        return x_q


@MODELS.register_module()
class DSOHead(BaseDecodeHead):
    """APFormerHead2 with DSO-BAMA attention (accuracy-first decoder).

    Compared to official APFormerHead2 (U-MixFormer):
      - Replace cross-attention with DSO-BAMA to integrate {fine KV} + {semantic agents} + {boundary agents}
        in a single softmax (no explicit gating).
      - Dual-source agents (Q + KV) and ortho-agent normalization.

    Args (important):
        feature_strides: strides of the 4 backbone stages (e.g., [4,8,16,32]).
        pool_ratio: how to downsample {s4,s3,s2,s1} to s4 resolution when building mixed KV.
                   Must be in order (s4->s1), typically (1,2,4,8).

        num_heads, agent_pool_sizes, kv_agent_pool_sizes are all stage-wise lists in order (s4->s1).

        boundary_pool_size: if None, boundary tokens are disabled. Otherwise, boundary tokens are pooled
                            from the shallowest feature map (c1).
    """

    def __init__(
        self,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        num_heads: Sequence[int] = (8, 5, 2, 1),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        qkv_bias: bool = False,
        # Agent tokens (semantic)
        agent_pool_sizes: Sequence[Union[int, Tuple[int, int]]] = ((7, 7), (7, 7), (7, 7), (7, 7)),
        kv_agent_pool_sizes: Sequence[Union[int, Tuple[int, int]]] = ((7, 7), (7, 7), (7, 7), (7, 7)),
        use_kv_agent: bool = True,
        use_learnable_agent: bool = False,
        use_ortho_agent: bool = True,
        ortho_eps: float = 1e-4,
        # Boundary tokens
        boundary_pool_size: Optional[Union[int, Tuple[int, int]]] = (7, 7),
        boundary_lowpass_kernel: int = 3,
        boundary_eps: float = 1e-6,
        boundary_use_abs: bool = True,
        # Local DWConv in attention / MLP
        use_dwconv_attn: bool = True,
        dwconv_attn_kernel_size: int = 3,
        dwconv_mlp_kernel_size: int = 3,
        # CatKey
        share_cat_key: bool = True,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        assert len(pool_ratio) == 4, 'pool_ratio must have 4 values for {s4,s3,s2,s1}'
        assert len(num_heads) == 4, 'num_heads must have 4 values for {s4,s3,s2,s1}'
        assert len(agent_pool_sizes) == 4
        assert len(kv_agent_pool_sizes) == 4

        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = int(c1_in + c2_in + c3_in + c4_in)

        # Boundary token pool (from c1)
        self.use_boundary = boundary_pool_size is not None
        if self.use_boundary:
            self.boundary_pool = EdgeAwareTokenPool(
                pool_size=boundary_pool_size,
                lowpass_kernel=boundary_lowpass_kernel,
                eps=boundary_eps,
                use_abs=boundary_use_abs,
            )
        else:
            self.boundary_pool = None

        # CatKey (KV builder)
        if share_cat_key:
            self.cat_key = CatKey(pool_ratio=pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
            self.cat_key1 = self.cat_key2 = self.cat_key3 = self.cat_key4 = self.cat_key
        else:
            self.cat_key1 = CatKey(pool_ratio=pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
            self.cat_key2 = CatKey(pool_ratio=pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
            self.cat_key3 = CatKey(pool_ratio=pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
            self.cat_key4 = CatKey(pool_ratio=pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])

        # One block per stage (s4->s1)
        # Note: dim_kv is always tot_channels because mixed KV concatenates all stage features.
        self.block_s4 = UMixFormerDSOBBlock(
            dim_q=c4_in, dim_kv=tot_channels, num_heads=num_heads[0],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            qkv_bias=qkv_bias,
            agent_pool_size=agent_pool_sizes[0],
            kv_agent_pool_size=kv_agent_pool_sizes[0],
            boundary_in_channels=c1_in,
            use_boundary=self.use_boundary,
            use_kv_agent=use_kv_agent,
            use_learnable_agent=use_learnable_agent,
            use_ortho_agent=use_ortho_agent,
            ortho_eps=ortho_eps,
            use_dwconv=use_dwconv_attn,
            dwconv_kernel_size=dwconv_attn_kernel_size,
            dwconv_mlp_kernel_size=dwconv_mlp_kernel_size,
        )
        self.block_s3 = UMixFormerDSOBBlock(
            dim_q=c3_in, dim_kv=tot_channels, num_heads=num_heads[1],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            qkv_bias=qkv_bias,
            agent_pool_size=agent_pool_sizes[1],
            kv_agent_pool_size=kv_agent_pool_sizes[1],
            boundary_in_channels=c1_in,
            use_boundary=self.use_boundary,
            use_kv_agent=use_kv_agent,
            use_learnable_agent=use_learnable_agent,
            use_ortho_agent=use_ortho_agent,
            ortho_eps=ortho_eps,
            use_dwconv=use_dwconv_attn,
            dwconv_kernel_size=dwconv_attn_kernel_size,
            dwconv_mlp_kernel_size=dwconv_mlp_kernel_size,
        )
        self.block_s2 = UMixFormerDSOBBlock(
            dim_q=c2_in, dim_kv=tot_channels, num_heads=num_heads[2],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            qkv_bias=qkv_bias,
            agent_pool_size=agent_pool_sizes[2],
            kv_agent_pool_size=kv_agent_pool_sizes[2],
            boundary_in_channels=c1_in,
            use_boundary=self.use_boundary,
            use_kv_agent=use_kv_agent,
            use_learnable_agent=use_learnable_agent,
            use_ortho_agent=use_ortho_agent,
            ortho_eps=ortho_eps,
            use_dwconv=use_dwconv_attn,
            dwconv_kernel_size=dwconv_attn_kernel_size,
            dwconv_mlp_kernel_size=dwconv_mlp_kernel_size,
        )
        self.block_s1 = UMixFormerDSOBBlock(
            dim_q=c1_in, dim_kv=tot_channels, num_heads=num_heads[3],
            mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            qkv_bias=qkv_bias,
            agent_pool_size=agent_pool_sizes[3],
            kv_agent_pool_size=kv_agent_pool_sizes[3],
            boundary_in_channels=c1_in,
            use_boundary=self.use_boundary,
            use_kv_agent=use_kv_agent,
            use_learnable_agent=use_learnable_agent,
            use_ortho_agent=use_ortho_agent,
            ortho_eps=ortho_eps,
            use_dwconv=use_dwconv_attn,
            dwconv_kernel_size=dwconv_attn_kernel_size,
            dwconv_mlp_kernel_size=dwconv_mlp_kernel_size,
        )

        # Fuse multi-stage outputs at c1 resolution
        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,  # use BaseDecodeHead.channels as embedding dim
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # (c1,c2,c3,c4)
        c1, c2, c3, c4 = x

        b, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # Boundary tokens from shallowest feature (once for all stages)
        boundary_tokens = None
        if self.boundary_pool is not None:
            boundary_tokens = self.boundary_pool(c1)  # [B,Nb,C1]

        # s4
        c_key = self.cat_key1([c4, c3, c2, c1]).flatten(2).transpose(1, 2).contiguous()
        c4_tok = c4.flatten(2).transpose(1, 2).contiguous()
        _c4_tok = self.block_s4(c4_tok, c_key, hw_q=(h4, w4), hw_kv=(h4, w4), boundary_tokens=boundary_tokens)
        _c4 = _c4_tok.transpose(1, 2).reshape(b, -1, h4, w4).contiguous()

        # s3 (replace c4 with decoded _c4)
        c_key = self.cat_key2([_c4, c3, c2, c1]).flatten(2).transpose(1, 2).contiguous()
        c3_tok = c3.flatten(2).transpose(1, 2).contiguous()
        _c3_tok = self.block_s3(c3_tok, c_key, hw_q=(h3, w3), hw_kv=(h4, w4), boundary_tokens=boundary_tokens)
        _c3 = _c3_tok.transpose(1, 2).reshape(b, -1, h3, w3).contiguous()

        # s2 (replace {c4,c3} with decoded {_c4,_c3})
        c_key = self.cat_key3([_c4, _c3, c2, c1]).flatten(2).transpose(1, 2).contiguous()
        c2_tok = c2.flatten(2).transpose(1, 2).contiguous()
        _c2_tok = self.block_s2(c2_tok, c_key, hw_q=(h2, w2), hw_kv=(h4, w4), boundary_tokens=boundary_tokens)
        _c2 = _c2_tok.transpose(1, 2).reshape(b, -1, h2, w2).contiguous()

        # s1 (replace {c4,c3,c2} with decoded {_c4,_c3,_c2})
        c_key = self.cat_key4([_c4, _c3, _c2, c1]).flatten(2).transpose(1, 2).contiguous()
        c1_tok = c1.flatten(2).transpose(1, 2).contiguous()
        _c1_tok = self.block_s1(c1_tok, c_key, hw_q=(h1, w1), hw_kv=(h4, w4), boundary_tokens=boundary_tokens)
        _c1 = _c1_tok.transpose(1, 2).reshape(b, -1, h1, w1).contiguous()

        # Upsample to c1 resolution and fuse
        _c4_up = resize(_c4, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        _c3_up = resize(_c3, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        _c2_up = resize(_c2, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        fused = self.linear_fuse(torch.cat([_c4_up, _c3_up, _c2_up, _c1], dim=1))
        out = self.cls_seg(fused)
        return out
