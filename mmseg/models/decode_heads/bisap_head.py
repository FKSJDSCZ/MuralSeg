# This file implements an improved U-MixFormer-style decoder head with
# Agent-Prior enhanced cross-attention.
#
# The design is intended to be dropped into `mmseg/models/decode_heads/`.

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

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


class DWConv(BaseModule):
    """Depthwise conv for token tensor (B, N, C) given spatial size."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(BaseModule):
    """FFN used by U-MixFormer head (Linear -> DWConv -> GELU -> Linear)."""

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

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CatKey(BaseModule):
    """Spatially align multi-stage features by pooling to the smallest map.

    Args:
        pool_ratio: list/tuple with length=4, in order (s4, s3, s2, s1).
        dim: list/tuple with length=4, channel dims for (s4, s3, s2, s1).

    Note:
        In U-MixFormer, (s4, s3, s2, s1) correspond to (c4, c3, c2, c1).
    """

    def __init__(
        self,
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        dim: Sequence[int] = (256, 160, 64, 32),
    ):
        super().__init__()
        assert len(pool_ratio) == len(dim) == 4
        self.pool_ratio = list(pool_ratio)

        self.sr_list = ModuleList(
            [nn.Conv2d(dim[i], dim[i], kernel_size=1, stride=1) for i in range(4) if self.pool_ratio[i] > 1],
        )
        self.pool_list = ModuleList(
            [nn.AvgPool2d(self.pool_ratio[i], self.pool_ratio[i], ceil_mode=True) for i in range(4) if
             self.pool_ratio[i] > 1],
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == 4
        out_list: List[torch.Tensor] = []
        cnt = 0
        for i in range(4):
            if self.pool_ratio[i] > 1:
                out_list.append(self.sr_list[cnt](self.pool_list[cnt](feats[i])))
                cnt += 1
            else:
                out_list.append(feats[i])
        return torch.cat(out_list, dim=1)


class AgentPriorCrossAttention(BaseModule):
    """Cross-attention enhanced by an agent-induced *probabilistic prior*.

    This module keeps the full-rank Softmax cross-attention as the main path,
    and injects an agent-based low-rank distribution Π(Q,K) as a log-prior
    into the attention logits.

    The prior is built with two Softmax operations (Agent Aggregation + Broadcast)
    akin to Agent Attention, but used as a *prior* (log Π) rather than the final output.

    Args:
        dim_q: Channel dim of query tokens.
        dim_kv: Channel dim of key/value tokens.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in q/kv projections.
        attn_drop, proj_drop: Dropout.
        use_agent_prior: Enable/disable the prior branch.
        agent_pool_size: (h, w) for adaptive pooling that generates n=h*w agent tokens.
        agent_from: 'kv' | 'q' | 'qkv'. Where to derive agent tokens.
        prior_strength: λ for logits += λ * log(Π + eps).
        prior_eps: epsilon for numerical stability.
        use_q_local_conv: Whether to add depthwise conv residual from query tokens.
        q_local_kernel: kernel size for local conv.
    """

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        use_agent_prior: bool = True,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        agent_from: str = 'kv',
        prior_strength: float = 1.0,
        prior_eps: float = 1e-6,
        use_q_local_conv: bool = False,
        q_local_kernel: int = 3,
    ):
        super().__init__()
        assert dim_q % num_heads == 0, f"dim_q {dim_q} should be divided by num_heads {num_heads}."
        assert agent_from in {'kv', 'q', 'qkv'}

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_agent_prior = use_agent_prior
        self.agent_from = agent_from
        self.prior_strength = float(prior_strength)
        self.prior_eps = float(prior_eps)

        if self.use_agent_prior:
            self.agent_pool_size = to_2tuple(agent_pool_size)
            self.agent_pool = nn.AdaptiveAvgPool2d(self.agent_pool_size)
            # Optional normalization when combining agents from q and kv.
            self.agent_norm = nn.LayerNorm(dim_q) if self.agent_from == 'qkv' else None
        else:
            self.agent_pool_size = (0, 0)
            self.agent_pool = None
            self.agent_norm = None

        self.use_q_local_conv = use_q_local_conv
        if self.use_q_local_conv:
            k = int(q_local_kernel)
            self.q_local = nn.Conv2d(dim_q, dim_q, kernel_size=k, stride=1, padding=k // 2, groups=dim_q)
        else:
            self.q_local = None

        self.softmax = nn.Softmax(dim=-1)

    def _pool_tokens(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Pool token tensor (B, N, C) -> (B, n_agents, C)."""
        assert self.agent_pool is not None
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        a_2d = self.agent_pool(x_2d)  # (B, C, Ah, Aw)
        a = a_2d.flatten(2).transpose(1, 2)  # (B, n_agents, C)
        return a

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        H_k: int,
        W_k: int,
        H_q: int,
        W_q: int,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: query tokens, shape (B, Nq, Cq).
            y: key/value tokens, shape (B, Nk, Ckv).
            H_k, W_k: spatial size of y.
            H_q, W_q: spatial size of x.
        """
        B, Nq, Cq = x.shape
        _, Nk, _ = y.shape

        q_lin = self.q(x)  # (B, Nq, Cq)
        kv_lin = self.kv(y)  # (B, Nk, 2*Cq)
        k_lin, v_lin = kv_lin[..., :Cq], kv_lin[..., Cq:]

        # reshape to heads
        q = q_lin.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, Nq, d)
        k = k_lin.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, Nk, d)
        v = v_lin.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, h, Nk, d)

        logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, Nq, Nk)

        if self.use_agent_prior:
            # Build agent tokens in the same feature space (dim_q).
            A_list = []
            if self.agent_from in {'q', 'qkv'}:
                A_list.append(self._pool_tokens(q_lin, H_q, W_q))  # (B, n, Cq)
            if self.agent_from in {'kv', 'qkv'}:
                A_list.append(self._pool_tokens(k_lin, H_k, W_k))  # (B, n, Cq)

            A = A_list[0]
            if len(A_list) == 2:
                A = A + A_list[1]  # direct sum (no gating)
                if self.agent_norm is not None:
                    A = self.agent_norm(A)

            # (B, h, n, d)
            n_agents = A.shape[1]
            A_h = A.reshape(B, n_agents, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # Π = softmax(QA^T) @ softmax(AK^T)
            p = self.softmax((q @ A_h.transpose(-2, -1)) * self.scale)  # (B, h, Nq, n)
            r = self.softmax((A_h @ k.transpose(-2, -1)) * self.scale)  # (B, h, n, Nk)
            prior = p @ r  # (B, h, Nq, Nk)

            # log-prior injection
            prior = prior.clamp(min=self.prior_eps)
            logits = logits + self.prior_strength * torch.log(prior)

        attn = self.softmax(logits)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Nq, Cq)

        # Optional local residual from query tokens (boundary/detail preservation).
        if self.use_q_local_conv and self.q_local is not None:
            x_local = q_lin.transpose(1, 2).reshape(B, Cq, H_q, W_q)
            x_local = self.q_local(x_local).flatten(2).transpose(1, 2)
            out = out + x_local

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class APXBlock(BaseModule):
    """Decoder block = (Norm -> AgentPriorCrossAttention) + (Norm -> MLP)."""

    def __init__(
        self,
        dim_q: int,
        dim_kv: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        *,
        use_agent_prior: bool = True,
        agent_pool_size: Union[int, Tuple[int, int]] = (7, 7),
        agent_from: str = 'kv',
        prior_strength: float = 1.0,
        prior_eps: float = 1e-6,
        use_q_local_conv: bool = False,
        q_local_kernel: int = 3,
    ):
        super().__init__()
        self.norm_q = norm_layer(dim_q)
        self.norm_kv = norm_layer(dim_kv)
        self.norm_post = norm_layer(dim_q)

        self.attn = AgentPriorCrossAttention(
            dim_q=dim_q,
            dim_kv=dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_agent_prior=use_agent_prior,
            agent_pool_size=agent_pool_size,
            agent_from=agent_from,
            prior_strength=prior_strength,
            prior_eps=prior_eps,
            use_q_local_conv=use_q_local_conv,
            q_local_kernel=q_local_kernel,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Mlp(in_features=dim_q, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        H_k: int,
        W_k: int,
        H_q: int,
        W_q: int,
    ) -> torch.Tensor:
        # Attention
        x = x + self.drop_path(self.attn(self.norm_q(x), self.norm_kv(y), H_k=H_k, W_k=W_k, H_q=H_q, W_q=W_q))
        # FFN
        x = x + self.drop_path(self.mlp(self.norm_post(x), H_q, W_q))
        return x


@MODELS.register_module()
class BiSAPHead(BaseDecodeHead):
    """U-MixFormer-style decoder head with Bi-Source Agent-Prior enhanced cross-attention.

    This head is a drop-in alternative to APFormerHead2.

    Notes on list argument order:
        All stage-wise lists are ordered as (s4, s3, s2, s1) from deep to shallow,
        i.e., corresponding to (c4, c3, c2, c1).

    Args:
        num_heads: attention heads per stage, (s4, s3, s2, s1).
        pool_ratio: pooling ratios for CatKey to align to the smallest map, (s4, s3, s2, s1).

        use_agent_prior: enable/disable agent prior branch in all stages.
        agent_pool_sizes: per-stage agent pooling grid sizes, list len=4 (s4..s1).
            If None, use `agent_pool_size` for all stages.
        agent_pool_size: default pooling size for agent tokens.
        agent_from: 'kv' | 'q' | 'qkv'.
        prior_strength: λ for logits prior.
        prior_eps: epsilon.

        use_q_local_conv: whether to add query-local DWConv residual in attention.
        q_local_kernels: per-stage kernel sizes list len=4 or scalar `q_local_kernel`.

        drop_path_rate: stochastic depth rate. If float, uses the same rate for all stages.

    All other kwargs are passed to BaseDecodeHead.
    """

    def __init__(
        self,
        num_heads: Sequence[int] = (8, 5, 2, 1),
        pool_ratio: Sequence[int] = (1, 2, 4, 8),
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: Union[float, Sequence[float]] = 0.1,
        # Agent-prior branch
        use_agent_prior: bool = True,
        agent_pool_sizes: Sequence[Union[int, Tuple[int, int]]] = (7, 7, 7, 7),
        agent_from: str = 'kv',
        prior_strength: float = 1.0,
        prior_eps: float = 1e-6,
        # Query-local conv inside attention
        use_q_local_conv: bool = False,
        q_local_kernels: Union[int, Sequence[int]] = 3,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        # Stage-wise parameters are in order (s4, s3, s2, s1) == (c4, c3, c2, c1)
        assert len(num_heads) == 4
        self.num_heads = list(num_heads)
        assert len(pool_ratio) == 4
        self.pool_ratio = list(pool_ratio)
        assert len(agent_pool_sizes) == 4
        self.agent_pool_sizes = list(agent_pool_sizes)
        self.q_local_kernels: Tuple = to_4tuple(q_local_kernels)

        dpr = to_4tuple(drop_path_rate)
        c1_in, c2_in, c3_in, c4_in = self.in_channels
        tot_channels = sum(self.in_channels)

        # Decoder blocks (s4->s1)
        self.dec_s4 = APXBlock(
            dim_q=c4_in,
            dim_kv=tot_channels,
            num_heads=self.num_heads[0],
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr[0],
            use_agent_prior=use_agent_prior,
            agent_pool_size=self.agent_pool_sizes[0],
            agent_from=agent_from,
            prior_strength=prior_strength,
            prior_eps=prior_eps,
            use_q_local_conv=use_q_local_conv,
            q_local_kernel=self.q_local_kernels[0],
        )
        self.dec_s3 = APXBlock(
            dim_q=c3_in,
            dim_kv=tot_channels,
            num_heads=self.num_heads[1],
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr[1],
            use_agent_prior=use_agent_prior,
            agent_pool_size=self.agent_pool_sizes[1],
            agent_from=agent_from,
            prior_strength=prior_strength,
            prior_eps=prior_eps,
            use_q_local_conv=use_q_local_conv,
            q_local_kernel=self.q_local_kernels[1],
        )
        self.dec_s2 = APXBlock(
            dim_q=c2_in,
            dim_kv=tot_channels,
            num_heads=self.num_heads[2],
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr[2],
            use_agent_prior=use_agent_prior,
            agent_pool_size=self.agent_pool_sizes[2],
            agent_from=agent_from,
            prior_strength=prior_strength,
            prior_eps=prior_eps,
            use_q_local_conv=use_q_local_conv,
            q_local_kernel=self.q_local_kernels[2],
        )
        self.dec_s1 = APXBlock(
            dim_q=c1_in,
            dim_kv=tot_channels,
            num_heads=self.num_heads[3],
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=dpr[3],
            use_agent_prior=use_agent_prior,
            agent_pool_size=self.agent_pool_sizes[3],
            agent_from=agent_from,
            prior_strength=prior_strength,
            prior_eps=prior_eps,
            use_q_local_conv=use_q_local_conv,
            q_local_kernel=self.q_local_kernels[3],
        )

        # Key/value mixing modules (one per stage, as in APFormerHead2)
        self.cat_key1 = CatKey(pool_ratio=self.pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key2 = CatKey(pool_ratio=self.pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key3 = CatKey(pool_ratio=self.pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])
        self.cat_key4 = CatKey(pool_ratio=self.pool_ratio, dim=[c4_in, c3_in, c2_in, c1_in])

        # Fusion + classifier
        self.linear_fuse = ConvModule(
            in_channels=tot_channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = self._transform_inputs(inputs)  # c1..c4
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape
        _, _, h1, w1 = c1.shape

        # Stage s4 (query=c4)
        kv = self.cat_key1([c4, c3, c2, c1])
        kv_tok = kv.flatten(2).transpose(1, 2)  # (B, Nk, Ckv)
        q4_tok = c4.flatten(2).transpose(1, 2)
        out4_tok = self.dec_s4(q4_tok, kv_tok, H_k=h4, W_k=w4, H_q=h4, W_q=w4)
        out4 = out4_tok.permute(0, 2, 1).reshape(n, -1, h4, w4)

        # Stage s3 (query=c3)
        kv = self.cat_key2([out4, c3, c2, c1])
        kv_tok = kv.flatten(2).transpose(1, 2)
        q3_tok = c3.flatten(2).transpose(1, 2)
        out3_tok = self.dec_s3(q3_tok, kv_tok, H_k=h4, W_k=w4, H_q=h3, W_q=w3)
        out3 = out3_tok.permute(0, 2, 1).reshape(n, -1, h3, w3)

        # Stage s2 (query=c2)
        kv = self.cat_key3([out4, out3, c2, c1])
        kv_tok = kv.flatten(2).transpose(1, 2)
        q2_tok = c2.flatten(2).transpose(1, 2)
        out2_tok = self.dec_s2(q2_tok, kv_tok, H_k=h4, W_k=w4, H_q=h2, W_q=w2)
        out2 = out2_tok.permute(0, 2, 1).reshape(n, -1, h2, w2)

        # Stage s1 (query=c1)
        kv = self.cat_key4([out4, out3, out2, c1])
        kv_tok = kv.flatten(2).transpose(1, 2)
        q1_tok = c1.flatten(2).transpose(1, 2)
        out1_tok = self.dec_s1(q1_tok, kv_tok, H_k=h4, W_k=w4, H_q=h1, W_q=w1)
        out1 = out1_tok.permute(0, 2, 1).reshape(n, -1, h1, w1)

        # Upsample to (h1, w1) and fuse
        out4_up = resize(out4, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        out3_up = resize(out3, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)
        out2_up = resize(out2, size=(h1, w1), mode='bilinear', align_corners=self.align_corners)

        fused = self.linear_fuse(torch.cat([out4_up, out3_up, out2_up, out1], dim=1))
        seg_logits = self.cls_seg(fused)
        return seg_logits
