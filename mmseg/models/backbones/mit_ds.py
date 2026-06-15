# This is a custom implementation for MuralSeg backbone.

import math
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_init

from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw

# Reuse official impl for disabled branches
from .mit import EfficientMultiheadAttention, MixFFN


def _as_stage_set(stages: Optional[Union[int, Sequence[Union[int, bool]]]],
                  num_stages: int) -> set:
    """Normalize stage selector to a set of stage indices."""
    if stages is None:
        return set()
    if isinstance(stages, int):
        return {int(stages)}
    if isinstance(stages, (list, tuple)):
        # bool-mask style: [True, False, True, ...]
        if len(stages) == num_stages and all(isinstance(x, bool) for x in stages):
            return {i for i, v in enumerate(stages) if v}
        # index list style: [0, 1, 2]
        return {int(s) for s in stages}
    raise TypeError(f'Unsupported stages type: {type(stages)}')


def _as_blocks_dict(blocks: Optional[Union[Dict[int, Sequence[int]], Sequence[Optional[Sequence[int]]]]],
                    num_stages: int) -> Optional[Dict[int, set]]:
    """Normalize blocks selector to dict: stage -> set(block_idx)."""
    if blocks is None:
        return None
    if isinstance(blocks, dict):
        out = {}
        for k, v in blocks.items():
            out[int(k)] = {int(i) for i in v}
        return out
    if isinstance(blocks, (list, tuple)):
        if len(blocks) != num_stages:
            raise ValueError(f'blocks as list/tuple must have length {num_stages}, '
                             f'but got {len(blocks)}')
        out = {}
        for s, idxs in enumerate(blocks):
            if idxs is None:
                continue
            out[s] = {int(i) for i in idxs}
        return out
    raise TypeError(f'Unsupported blocks type: {type(blocks)}')


def _build_enable_map(cfg: Optional[dict], num_stages: int,
                      num_layers: Sequence[int]) -> List[List[bool]]:
    """Return enable map: enable_map[stage][block]."""
    enable_map: List[List[bool]] = [[False] * int(n) for n in num_layers]
    if cfg is None:
        return enable_map

    if not cfg.get('enable', True):
        return enable_map

    stages = _as_stage_set(cfg.get('stages', tuple(range(num_stages))), num_stages)
    blocks_dict = _as_blocks_dict(cfg.get('blocks', None), num_stages)

    for s in range(num_stages):
        if s not in stages:
            continue
        if blocks_dict is None or s not in blocks_dict:
            enable_map[s] = [True] * int(num_layers[s])
        else:
            for b in blocks_dict[s]:
                if 0 <= b < int(num_layers[s]):
                    enable_map[s][b] = True
    return enable_map


def _get_stage_value(cfg: Optional[dict], key: str, stage: int, default):
    """Allow cfg[key] to be scalar or dict(stage->value)."""
    if cfg is None:
        return default
    val = cfg.get(key, default)
    if isinstance(val, dict):
        return val.get(stage, default)
    return val


@MODELS.register_module()
class EPSREfficientMultiheadAttention(EfficientMultiheadAttention):
    """Efficient MHA with Edge-Preserved Spatial-Reduction (EPSR).

    Only active when sr_ratio > 1 and epsr_enable=True.
    """

    def __init__(self,
                 *args,
                 epsr_enable: bool = True,
                 epsr_alpha_init: float = 1.0,
                 epsr_dwconv_kernel_size: int = 3,
                 **kwargs):
        self.epsr_enable = bool(epsr_enable)
        super().__init__(*args, **kwargs)

        if self.epsr_enable and self.sr_ratio > 1:
            k = int(epsr_dwconv_kernel_size)
            assert k % 2 == 1, 'epsr_dwconv_kernel_size should be odd.'
            self.epsr_dwconv = Conv2d(
                in_channels=self.embed_dims,
                out_channels=self.embed_dims,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                groups=self.embed_dims,
                bias=True)
            self.epsr_alpha = nn.Parameter(torch.tensor(float(epsr_alpha_init)))
        else:
            self.epsr_dwconv = None
            self.epsr_alpha = None

    def _epsr_reduce(self, x: torch.Tensor, hw_shape: Tuple[int, int]) -> torch.Tensor:
        """x: (B, L, C) -> EPSR -> SR -> (B, L', C)."""
        # x -> BCHW
        x_nchw = nlc_to_nchw(x, hw_shape)  # (B, C, H, W)
        # edge map
        m = self.epsr_dwconv(x_nchw).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        m = torch.sigmoid(m)
        # enhance then SR
        x_enh = x_nchw * (1.0 + self.epsr_alpha * m)
        x_sr = self.sr(x_enh)
        x_sr = nchw_to_nlc(x_sr)
        x_sr = self.norm(x_sr)
        return x_sr

    def forward(self, x, hw_shape, identity=None):
        # fallback to official when disabled / sr_ratio == 1
        if (not self.epsr_enable) or (self.sr_ratio <= 1):
            return super().forward(x, hw_shape, identity=identity)

        x_q = x
        x_kv = self._epsr_reduce(x, hw_shape)

        if identity is None:
            identity = x_q

        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        # legacy path for old mmcv: keep EPSR behavior
        if (not self.epsr_enable) or (self.sr_ratio <= 1):
            return super().legacy_forward(x, hw_shape, identity=identity)

        x_q = x
        x_kv = self._epsr_reduce(x, hw_shape)

        if identity is None:
            identity = x_q

        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]
        return identity + self.dropout_layer(self.proj_drop(out))


@MODELS.register_module()
class MSDMixFFN(BaseModule):
    """Multi-Scale Dilated MixFFN (MSD-MixFFN).

    Design goals:
    1) Keep compatibility for important pretrained weights:
       - fc1, base pe_conv(dilation=1), fc2 positions follow official MixFFN `layers`.
    2) Add extra dilated depthwise conv branches + dynamic fusion.
    """

    def __init__(self,
                 embed_dims: int,
                 feedforward_channels: int,
                 dilations: Sequence[int] = (1, 2, 3),
                 act_cfg: dict = dict(type='GELU'),
                 ffn_drop: float = 0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = int(embed_dims)
        self.feedforward_channels = int(feedforward_channels)

        # normalize dilations: ensure 1 is first (for pretrained mapping)
        dil_list = list(dilations)
        dil_list = [int(d) for d in dil_list]
        if 1 in dil_list:
            dil_list = [1] + [d for d in dil_list if d != 1]
        else:
            dil_list = [1] + dil_list
        # remove duplicates while preserving order
        seen = set()
        self.dilations = []
        for d in dil_list:
            if d not in seen:
                self.dilations.append(d)
                seen.add(d)
        self.num_branches = len(self.dilations)

        self.activate = build_activation_layer(act_cfg)

        # keep official MixFFN naming style for partial pretrained loading
        fc1 = Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        # base dilation=1 depthwise conv (same as official pe_conv)
        pe_conv = Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=True,
            groups=self.feedforward_channels)

        fc2 = Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.embed_dims,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)

        # layers index mapping:
        # layers[0]=fc1, layers[1]=base pe_conv(d=1), layers[2]=act, layers[3]=drop,
        # layers[4]=fc2, layers[5]=drop
        self.layers = Sequential(fc1, pe_conv, self.activate, drop, fc2, drop)

        # extra dilated depthwise conv branches (d>1)
        extra = []
        for d in self.dilations[1:]:
            extra.append(
                Conv2d(
                    in_channels=self.feedforward_channels,
                    out_channels=self.feedforward_channels,
                    kernel_size=3,
                    stride=1,
                    padding=d,
                    dilation=d,
                    bias=True,
                    groups=self.feedforward_channels))
        self.extra_pe_convs = nn.ModuleList(extra)

        # dynamic fusion weights: GAP -> Linear -> softmax
        self.fuse_fc = nn.Linear(self.feedforward_channels, self.num_branches)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        # x: (B, L, C)
        out = nlc_to_nchw(x, hw_shape)  # (B, C, H, W)

        # fc1
        z = self.layers[0](out)  # (B, Cffn, H, W)

        # multi-branch depthwise conv
        branch_outs = []
        branch_outs.append(self.layers[1](z))  # base dilation=1 conv
        for conv in self.extra_pe_convs:
            branch_outs.append(conv(z))

        # fusion weights
        gap = z.mean(dim=(2, 3))  # (B, Cffn)
        beta = torch.softmax(self.fuse_fc(gap), dim=1)  # (B, K)

        fused = 0.0
        for i, bo in enumerate(branch_outs):
            fused = fused + bo * beta[:, i].view(-1, 1, 1, 1)

        # act, drop, fc2, drop (reuse official indices)
        fused = self.layers[2](fused)
        fused = self.layers[3](fused)
        fused = self.layers[4](fused)
        fused = self.layers[5](fused)

        fused = nchw_to_nlc(fused)  # (B, L, C)

        if identity is None:
            identity = x
        return identity + self.dropout_layer(fused)


class DSTransformerEncoderLayer(BaseModule):
    """One encoder block with optional EPSR-Attn and MSD-MixFFN."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 batch_first: bool = True,
                 sr_ratio: int = 1,
                 with_cp: bool = False,
                 # switches
                 use_epsr_attn: bool = False,
                 epsr_kwargs: Optional[dict] = None,
                 use_msd_ffn: bool = False,
                 msd_kwargs: Optional[dict] = None):
        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        epsr_kwargs = epsr_kwargs or {}
        if use_epsr_attn:
            self.attn = EPSREfficientMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                batch_first=batch_first,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                sr_ratio=sr_ratio,
                **epsr_kwargs)
        else:
            self.attn = EfficientMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                batch_first=batch_first,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                sr_ratio=sr_ratio)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        msd_kwargs = msd_kwargs or {}
        if use_msd_ffn:
            self.ffn = MSDMixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                act_cfg=act_cfg,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                **msd_kwargs)
        else:
            self.ffn = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                act_cfg=act_cfg,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(_x):
            _x = self.attn(self.norm1(_x), hw_shape, identity=_x)
            _x = self.ffn(self.norm2(_x), hw_shape, identity=_x)
            return _x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class DSMixVisionTransformer(BaseModule):
    """MuralSeg backbone (MiT) with switchable EPSR-Attn / MSD-MixFFN.

    Args:
        epsr_cfg (dict|None):
            - enable (bool): default True
            - stages (Sequence[int] | Sequence[bool]): which stages enable EPSR
            - blocks (dict|list|None): optional, per-stage block indices
            - alpha_init (float|dict): scalar or {stage: value}
            - dwconv_kernel_size (int|dict): scalar or {stage: value}
        msd_cfg (dict|None):
            - enable (bool): default True
            - stages / blocks: same meaning
            - dilations (Sequence[int] | dict): scalar list or {stage: list}
    """

    def __init__(self,
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
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 pretrained: Optional[str] = None,
                 init_cfg=None,
                 with_cp: bool = False,
                 # NEW
                 epsr_cfg: Optional[dict] = None,
                 msd_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
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

        assert num_stages == len(self.num_layers) == len(self.num_heads) \
               == len(self.patch_sizes) == len(self.strides) == len(self.sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # store cfg
        self.epsr_cfg = deepcopy(epsr_cfg) if epsr_cfg is not None else None
        self.msd_cfg = deepcopy(msd_cfg) if msd_cfg is not None else None

        epsr_enable_map = _build_enable_map(self.epsr_cfg, num_stages, self.num_layers)
        msd_enable_map = _build_enable_map(self.msd_cfg, num_stages, self.num_layers)

        # stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.num_layers))]

        cur = 0
        self.layers = ModuleList()
        for stage_idx, depth in enumerate(self.num_layers):
            embed_dims_i = embed_dims * self.num_heads[stage_idx]

            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=self.patch_sizes[stage_idx],
                stride=self.strides[stage_idx],
                padding=self.patch_sizes[stage_idx] // 2,
                norm_cfg=norm_cfg)

            # per-stage epsr params (allow dict(stage->value))
            epsr_kwargs = {}
            if self.epsr_cfg is not None:
                epsr_kwargs = dict(
                    epsr_enable=True,
                    epsr_alpha_init=_get_stage_value(self.epsr_cfg, 'alpha_init', stage_idx, 1.0),
                    epsr_dwconv_kernel_size=_get_stage_value(self.epsr_cfg, 'dwconv_kernel_size', stage_idx, 3),
                )

            # per-stage msd params
            msd_kwargs = {}
            if self.msd_cfg is not None:
                msd_kwargs = dict(
                    dilations=_get_stage_value(self.msd_cfg, 'dilations', stage_idx, (1, 2, 3))
                )

            blocks = ModuleList()
            for block_idx in range(depth):
                blocks.append(
                    DSTransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=self.num_heads[stage_idx],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[cur + block_idx],
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        sr_ratio=self.sr_ratios[stage_idx],
                        use_epsr_attn=epsr_enable_map[stage_idx][block_idx],
                        epsr_kwargs=epsr_kwargs,
                        use_msd_ffn=msd_enable_map[stage_idx][block_idx],
                        msd_kwargs=msd_kwargs
                    )
                )

            in_channels = embed_dims_i
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]

            self.layers.append(ModuleList([patch_embed, blocks, norm]))
            cur += depth

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)  # PatchEmbed -> (B, L, C), (H, W)
            for blk in layer[1]:
                x = blk(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs
