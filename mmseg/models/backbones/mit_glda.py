"""MuralSeg backbones for mmsegmentation.

This file provides three MiT-style backbones for ablation:
- MixVisionTransformerGlda:      baseline + GLDA Block (ESA + LocalDense fusion)
- MixVisionTransformerTmx:      baseline + TextureMix-FFN
- MixVisionTransformerGldaTmx:  baseline + GLDA Block + TextureMix-FFN
`
Design goals:
- Keep SegFormer Efficient Self-Attention (ESA) as the global branch.
- Fuse ESA with a local dense window attention branch (WDSA) using a gate.
- Keep the rest of MiT-B{0..5} design (overlap patch embedding, stage-wise SR ratios).
- Code reuse: all 3 backbones share the same implementation with thin wrappers.

Note
----
This module is intended to be imported via mmseg's `custom_imports` mechanism.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import Conv2d, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_, trunc_normal_init

from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# Reuse official SegFormer ESA implementation.
from .mit import EfficientMultiheadAttention, MixFFN


def _to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Partition feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size: window size (M)

    Returns:
        windows: (B*nW, M*M, C)
        meta: (Hp, Wp, pad_h, pad_w)
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows, (Hp, Wp, pad_h, pad_w)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    meta: Tuple[int, int, int, int],
    original_hw: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    """Reverse window_partition.

    Args:
        windows: (B*nW, M*M, C)
        window_size: M
        meta: (Hp, Wp, pad_h, pad_w)
        original_hw: (H, W)
        batch_size: B

    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp, pad_h, pad_w = meta
    H, W = original_hw

    x = windows.view(
        batch_size,
        Hp // window_size,
        Wp // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, Hp, Wp, -1)

    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :].contiguous()
    return x


class RelativePositionBias(BaseModule):
    """2D relative position bias for window attention (Swin-style)."""

    def __init__(self, window_size: int, num_heads: int, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.window_size = window_size
        self.num_heads = num_heads

        # (2M-1)*(2M-1), nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, M, M)
        coords_flatten = torch.flatten(coords, 1)  # (2, M*M)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, M*M, M*M)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (M*M, M*M, 2)
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (M*M, M*M)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        """Return relative position bias.

        Returns:
            bias: (num_heads, M*M, M*M)
        """
        table = self.relative_position_bias_table
        index = self.relative_position_index.view(-1)
        bias = table[index].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )  # (M*M, M*M, nH)
        bias = bias.permute(2, 0, 1).contiguous()  # (nH, M*M, M*M)
        return bias


class WindowDenseSelfAttention(BaseModule):
    """Local dense window self-attention (WDSA).

    Input/Output token format is NLC (B, N, C). `hw_shape` is required to
    reshape into windows.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert embed_dims % num_heads == 0, 'embed_dims must be divisible by num_heads'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dims // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos = use_rel_pos
        self.rel_pos = RelativePositionBias(window_size, num_heads) if use_rel_pos else None

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        B, N, C = x.shape
        H, W = hw_shape
        assert N == H * W, f'Expected N==H*W, but got N={N}, H*W={H * W}'

        x_2d = x.view(B, H, W, C)
        x_windows, meta = window_partition(x_2d, self.window_size)  # (B*nW, M*M, C)

        BnW, Nw, _ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(BnW, Nw, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, BnW, nH, Nw, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BnW, nH, Nw, Nw)
        if self.use_rel_pos:
            attn = attn + self.rel_pos().unsqueeze(0)  # broadcast to (1,nH,Nw,Nw)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(BnW, Nw, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out_2d = window_unpartition(out, self.window_size, meta, (H, W), batch_size=B)
        out = out_2d.view(B, N, C)
        return out


class GlobalLocalDualAttention(BaseModule):
    """GLDA = ESA (global) + WDSA (local) + gated fusion.

    This module returns the *update* (no residual add). Residual + DropPath is
    handled in the encoder layer.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int,
        sr_ratio: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gate_type: str = 'channel',  # 'channel' or 'scalar'
        use_rel_pos: bool = True,
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims

        # Global branch: keep official ESA (spatial reduction attention).
        # - We disable its DropPath here and apply DropPath after fusion.
        self.esa = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            dropout_layer=None,
            batch_first=True,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio,
        )

        # Local branch: dense window attention.
        self.wdsa = WindowDenseSelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos=use_rel_pos,
        )

        # Fusion gate.
        assert gate_type in {'channel', 'scalar'}, f'gate_type must be "channel" or "scalar", got {gate_type!r}'
        self.gate_type = gate_type
        out_gate_dim = embed_dims if gate_type == 'channel' else 1
        self.gate_proj = nn.Linear(embed_dims * 2, out_gate_dim)

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        # x is assumed already normalized (pre-norm) in the encoder layer.
        local_update = self.wdsa(x, hw_shape)
        # Trick: pass identity=0 to avoid allocating a zeros-like tensor.
        global_update = self.esa(x, hw_shape, identity=0)

        gate_in = torch.cat([local_update, global_update], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_in))
        if self.gate_type == 'scalar':
            gate = gate.expand_as(local_update)

        fused_update = gate * local_update + (1.0 - gate) * global_update
        return fused_update


class TextureMixFFN(BaseModule):
    """TextureMix-FFN: multi-branch depthwise conv mixing after fc1.

    The design follows the proposal:
    - 3×3 DWConv
    - 3×3 DWConv (dilation=2)
    - 5×5 DWConv
    with softmax-adaptive weights.

    Input/Output: NLC.
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        ffn_drop: float = 0.0,
        dropout_layer: Optional[dict] = None,
        act_cfg: dict = dict(type='GELU'),
        init_cfg=None,
        # TextureMix specific
        gate_reduction: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.eps = eps

        # We implement the MiT-style MixFFN using 1x1 convs on NCHW
        self.fc1 = Conv2d(embed_dims, feedforward_channels, kernel_size=1, bias=True)
        # multi-scale depthwise convs
        self.dw3 = Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=3,
            padding=1,
            groups=feedforward_channels,
            bias=True,
        )
        self.dw3_d2 = Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=feedforward_channels,
            bias=True,
        )
        self.dw5 = Conv2d(
            feedforward_channels,
            feedforward_channels,
            kernel_size=5,
            padding=2,
            groups=feedforward_channels,
            bias=True,
        )

        # adaptive weight generator (per-sample weights for 3 branches)
        hidden = max(feedforward_channels // gate_reduction, 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feedforward_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

        # activation + dropout
        # Use mmcv activation builder for consistency? Here we keep it simple.
        act_type = act_cfg.get('type', 'GELU')
        if act_type.lower() == 'gelu':
            self.activate = nn.GELU()
        elif act_type.lower() == 'relu':
            self.activate = nn.ReLU(inplace=act_cfg.get('inplace', True))
        else:
            raise ValueError(f'Unsupported act_cfg for TextureMixFFN: {act_cfg}')

        self.drop = nn.Dropout(ffn_drop)
        self.fc2 = Conv2d(feedforward_channels, embed_dims, kernel_size=1, bias=True)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int],
                identity: Optional[torch.Tensor] = None) -> torch.Tensor:
        if identity is None:
            identity = x

        x = nlc_to_nchw(x, hw_shape)  # (B, C, H, W)
        x = self.fc1(x)

        # weights
        pooled = x.mean(dim=(2, 3))  # (B, ffn)
        a = torch.softmax(self.gate_mlp(pooled), dim=-1)  # (B, 3)
        a0 = a[:, 0].view(-1, 1, 1, 1)
        a1 = a[:, 1].view(-1, 1, 1, 1)
        a2 = a[:, 2].view(-1, 1, 1, 1)

        x_mix = a0 * self.dw3(x) + a1 * self.dw3_d2(x) + a2 * self.dw5(x)

        x_mix = self.activate(x_mix)
        x_mix = self.drop(x_mix)
        x_mix = self.fc2(x_mix)
        x_mix = self.drop(x_mix)
        x_mix = nchw_to_nlc(x_mix)

        return identity + self.dropout_layer(x_mix)


class TextureMixTransformerEncoderLayer(BaseModule):
    """Encoder layer = ESA + TextureMix-FFN (no GLDA fusion)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        batch_first: bool = True,
        sr_ratio: int = 1,
        with_cp: bool = False,
    ):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio,
        )
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = TextureMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
        )
        self.with_cp = with_cp

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        def _inner_forward(_x: torch.Tensor) -> torch.Tensor:
            _x = self.attn(self.norm1(_x), hw_shape, identity=_x)
            _x = self.ffn(self.norm2(_x), hw_shape, identity=_x)
            return _x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


class GLDATransformerEncoderLayer(BaseModule):
    """Encoder layer = GLDA + (MixFFN or TextureMixFFN)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        window_size: int,
        sr_ratio: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        with_cp: bool = False,
        # ffn selection
        ffn_type: str = 'mix',  # 'mix' or 'texture_mix'
        gate_type: str = 'channel',
        use_rel_pos: bool = True,
    ):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = GlobalLocalDualAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            gate_type=gate_type,
            use_rel_pos=use_rel_pos,
            norm_cfg=norm_cfg,
        )
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0 else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        assert ffn_type in {'mix', 'texture_mix'}, f'ffn_type must be "mix" or "texture_mix", got {ffn_type!r}'
        self.ffn_type = ffn_type
        if ffn_type == 'mix':
            self.ffn = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
            )
        else:
            self.ffn = TextureMixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
            )

        self.with_cp = with_cp

    def forward(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:

        def _inner_forward(_x: torch.Tensor) -> torch.Tensor:
            # attention update (no residual inside GLDA)
            update = self.attn(self.norm1(_x), hw_shape)
            _x = _x + self.drop_path(update)
            _x = self.ffn(self.norm2(_x), hw_shape, identity=_x)
            return _x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


class _MixVisionTransformerCustom(BaseModule):
    """A MiT backbone builder that can swap the encoder layer class."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 64,
        num_stages: int = 4,
        num_layers: Sequence[int] = (3, 4, 6, 3),
        num_heads: Sequence[int] = (1, 2, 4, 8),
        patch_sizes: Sequence[int] = (7, 3, 3, 3),
        strides: Sequence[int] = (4, 2, 2, 2),
        sr_ratios: Sequence[int] = (8, 4, 2, 1),
        out_indices: Sequence[int] = (0, 1, 2, 3),
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict = dict(type='GELU'),
        norm_cfg: dict = dict(type='LN', eps=1e-6),
        pretrained: Optional[str] = None,
        init_cfg: Optional[dict] = None,
        with_cp: bool = False,
        # custom
        local_window_sizes: Union[int, Sequence[int]] = 7,
        layer_type: str = 'baseline',  # baseline | glda | tmx | glda_tmx
        gate_type: str = 'channel',
        use_rel_pos: bool = True,
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            # keep compatibility with older configs
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = list(num_layers)
        self.num_heads = list(num_heads)
        self.patch_sizes = list(patch_sizes)
        self.strides = list(strides)
        self.sr_ratios = list(sr_ratios)
        self.with_cp = with_cp

        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        if isinstance(local_window_sizes, int):
            local_window_sizes = [local_window_sizes] * num_stages
        else:
            local_window_sizes = list(local_window_sizes)
            assert len(local_window_sizes) == num_stages
        self.local_window_sizes = local_window_sizes

        self.layer_type = layer_type

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]

        cur = 0
        self.layers = ModuleList()
        for stage_idx, depth in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[stage_idx]

            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[stage_idx],
                stride=strides[stage_idx],
                padding=patch_sizes[stage_idx] // 2,
                norm_cfg=norm_cfg,
            )

            blocks = ModuleList()
            for layer_idx in range(depth):
                dp = dpr[cur + layer_idx]
                if layer_type == 'tmx':
                    block = TextureMixTransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[stage_idx],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dp,
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        sr_ratio=sr_ratios[stage_idx],
                    )
                elif layer_type == 'glda':
                    block = GLDATransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[stage_idx],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        window_size=local_window_sizes[stage_idx],
                        sr_ratio=sr_ratios[stage_idx],
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dp,
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        ffn_type='mix',
                        gate_type=gate_type,
                        use_rel_pos=use_rel_pos,
                    )
                elif layer_type == 'glda_tmx':
                    block = GLDATransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[stage_idx],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        window_size=local_window_sizes[stage_idx],
                        sr_ratio=sr_ratios[stage_idx],
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dp,
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        ffn_type='texture_mix',
                        gate_type=gate_type,
                        use_rel_pos=use_rel_pos,
                    )
                else:
                    raise ValueError(
                        'This internal builder expects layer_type in {tmx, glda, glda_tmx}. '
                        f'Got {layer_type!r}. For baseline, please use mmseg.models.backbones.mit.MixVisionTransformer.'
                    )

                blocks.append(block)

            in_channels = embed_dims_i
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, blocks, norm]))
            cur += depth

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.0)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0.0, std=math.sqrt(2.0 / fan_out), bias=0.0)
        else:
            super().init_weights()

        for m in self.modules():
            if isinstance(m, GlobalLocalDualAttention):
                nn.init.zeros_(m.gate_proj.weight)
                nn.init.constant_(m.gate_proj.bias, -2.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            patch_embed, blocks, norm = layer
            x, hw_shape = patch_embed(x)
            for blk in blocks:
                x = blk(x, hw_shape)
            x = norm(x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs


@MODELS.register_module()
class MixVisionTransformerGlda(_MixVisionTransformerCustom):
    """MiT backbone: baseline + GLDA Block (ESA + LocalDense fusion)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, layer_type='glda', **kwargs)


@MODELS.register_module()
class MixVisionTransformerTmx(_MixVisionTransformerCustom):
    """MiT backbone: baseline + TextureMix-FFN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, layer_type='tmx', **kwargs)


@MODELS.register_module()
class MixVisionTransformerGldaTmx(_MixVisionTransformerCustom):
    """MiT backbone: baseline + GLDA Block + TextureMix-FFN."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, layer_type='glda_tmx', **kwargs)
