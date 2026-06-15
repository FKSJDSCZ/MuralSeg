"""PSDR decode head with optional deep supervision.

PSDR = Progressive Semantic-to-Detail Refinement decoder (coarse-to-fine),
following the uploaded design:
- D4: Coarse logits from F4
- D3/D2/D1: refine with UFF blocks using (Fs, Up(S_{s+1}))

UFF (Uncertainty-guided Feature Fusion):
- Uncertainty map U from entropy of softmax(Up(logits))
- Semantic query E from class distribution P via a 1x1 projection
- Local window cross-attention: query(E) attends to encoder features Fs within windows
- Uncertainty gating to focus refinement
- Residual logits refinement: S_s = Up(S_{s+1}) + DeltaS_s

This implementation is designed for mmsegmentation v1.x (mmengine-based).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize


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
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

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

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index, persistent=False)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self) -> torch.Tensor:
        table = self.relative_position_bias_table
        index = self.relative_position_index.view(-1)
        bias = table[index].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        bias = bias.permute(2, 0, 1).contiguous()  # (nH, Nw, Nw)
        return bias


class WindowCrossAttention(BaseModule):
    """Local window cross-attention.

    Query comes from q_map, key/value come from kv_map.

    Args:
        dim: channel dimension
        num_heads: number of attention heads
        window_size: local window size
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos = use_rel_pos
        self.rel_pos = RelativePositionBias(window_size, num_heads) if use_rel_pos else None

    def forward(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            q_map: (B, C, H, W)
            kv_map: (B, C, H, W)

        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = q_map.shape
        assert kv_map.shape[:3] == (B, C, H) and kv_map.shape[3] == W

        q_bhwc = q_map.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        kv_bhwc = kv_map.permute(0, 2, 3, 1).contiguous()

        q_windows, meta = window_partition(q_bhwc, self.window_size)  # (BnW, Nw, C)
        kv_windows, _ = window_partition(kv_bhwc, self.window_size)

        BnW, Nw, _ = q_windows.shape

        q = self.q(q_windows).reshape(BnW, Nw, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(kv_windows).reshape(BnW, Nw, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (BnW,nH,Nw,Nw)
        if self.use_rel_pos:
            attn = attn + self.rel_pos().unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(BnW, Nw, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out_bhwc = window_unpartition(out, self.window_size, meta, (H, W), batch_size=B)
        out = out_bhwc.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        return out


class SegLogitsHead(BaseModule):
    """A lightweight prediction head for (deep-supervised) logits."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_ratio: float = 0.0,
        norm_cfg: Optional[dict] = None,
        act_cfg: dict = dict(type='ReLU', inplace=True),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.dropout_ratio = dropout_ratio
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        # optional 3x3 refinement before logits (helps stabilize deep sup)
        self.refine = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)
        x = self.dropout(x)
        return self.conv_seg(x)


class UFFBlock(BaseModule):
    """UFF refinement block (coarse-to-fine)."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        window_size: int = 7,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        lambda_u: float = 1.0,
        eps: float = 1e-6,
        use_rel_pos: bool = True,
        align_corners: bool = False,
        norm_cfg: Optional[dict] = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=True),
        dropout_ratio: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.eps = eps
        self.align_corners = align_corners

        # semantic query projection: E = P W_e
        self.semantic_proj = nn.Conv2d(num_classes, channels, kernel_size=1, bias=False)

        # local window cross-attention
        self.cross_attn = WindowCrossAttention(
            dim=channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_rel_pos=use_rel_pos,
        )

        # fuse [F_s, O', Up(S_{s+1})] -> feature
        self.fuse1 = ConvModule(
            channels * 2 + num_classes,
            channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.fuse2 = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.delta_head = SegLogitsHead(
            in_channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    @staticmethod
    def _entropy_from_prob(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # p: (B, C, H, W)
        return -(p * (p.clamp_min(eps)).log()).sum(dim=1, keepdim=True)

    def forward(self, feat: torch.Tensor, prev_logits: torch.Tensor) -> torch.Tensor:
        """Refine logits at current scale.

        Args:
            feat: (B, channels, H, W)
            prev_logits: (B, num_classes, H_prev, W_prev)

        Returns:
            logits: (B, num_classes, H, W)
        """
        B, C, H, W = feat.shape
        prev_up = resize(prev_logits, size=(H, W), mode='bilinear', align_corners=self.align_corners)

        p = F.softmax(prev_up, dim=1)
        u = self._entropy_from_prob(p, eps=self.eps)  # (B,1,H,W)

        e = self.semantic_proj(p)  # (B,channels,H,W)
        o = self.cross_attn(e, feat)  # (B,channels,H,W)

        # uncertainty gating
        o = (1.0 + self.lambda_u * u) * o

        fused = torch.cat([feat, o, prev_up], dim=1)
        fused = self.fuse1(fused)
        fused = self.fuse2(fused)

        delta = self.delta_head(fused)
        logits = prev_up + delta
        return logits


@dataclass
class DeepSupervisionCfg:
    enable: bool
    keys: Tuple[str, ...]
    weights: Dict[str, float]


def _parse_deep_supervision_cfg(deep_supervision: Optional[Union[Dict, Sequence[str]]]) -> DeepSupervisionCfg:
    if deep_supervision is None or deep_supervision is False:
        return DeepSupervisionCfg(enable=False, keys=tuple(), weights={})

    if isinstance(deep_supervision, (list, tuple)):
        keys = tuple(str(k) for k in deep_supervision)
        return DeepSupervisionCfg(enable=len(keys) > 0, keys=keys, weights={k: 1.0 for k in keys})

    if isinstance(deep_supervision, dict):
        enable = bool(deep_supervision.get('enable', True))
        keys = tuple(deep_supervision.get('keys', ()))
        weights_in = deep_supervision.get('weights', None)
        if weights_in is None:
            # default: 0.4 for every aux output
            weights = {str(k): float(deep_supervision.get('default_weight', 0.4)) for k in keys}
        elif isinstance(weights_in, dict):
            weights = {str(k): float(v) for k, v in weights_in.items()}
        else:
            # list/tuple aligned with keys
            w_list = list(weights_in)
            assert len(w_list) == len(keys), 'deep_supervision.weights must align with keys'
            weights = {str(k): float(w) for k, w in zip(keys, w_list)}

        return DeepSupervisionCfg(enable=enable and len(keys) > 0, keys=tuple(str(k) for k in keys), weights=weights)

    raise TypeError('deep_supervision must be None/False, a sequence of keys, or a dict')


@MODELS.register_module()
class PSDRHead(BaseDecodeHead):
    """PSDR decode head with UFF refinement and optional deep supervision."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int,
        num_classes: int,
        in_index: Sequence[int] = (0, 1, 2, 3),
        input_transform: str = 'multiple_select',
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        # PSDR/UFF
        window_size: int = 7,
        num_heads: int = 4,
        lambda_u: float = 1.0,
        use_rel_pos: bool = True,
        # deep supervision
        deep_supervision: Optional[Union[Dict, Sequence[str]]] = None,
        ds_dropout_ratio: float = 0.0,
        norm_cfg: Optional[dict] = dict(type='BN', requires_grad=True),
        act_cfg: dict = dict(type='ReLU', inplace=True),
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            in_index=in_index,
            input_transform=input_transform,
            dropout_ratio=dropout_ratio,
            align_corners=align_corners,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.ds_cfg = _parse_deep_supervision_cfg(deep_supervision)

        # stage projections to a unified channels
        self.proj_s1 = ConvModule(in_channels[0], channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj_s2 = ConvModule(in_channels[1], channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj_s3 = ConvModule(in_channels[2], channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj_s4 = ConvModule(in_channels[3], channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # D4 coarse head
        self.coarse_conv = ConvModule(channels, channels, kernel_size=3, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.coarse_pred = SegLogitsHead(
            in_channels=channels,
            num_classes=num_classes,
            dropout_ratio=ds_dropout_ratio,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # D3/D2/D1 refinement blocks
        self.uff3 = UFFBlock(
            channels=channels,
            num_classes=num_classes,
            window_size=window_size,
            num_heads=num_heads,
            lambda_u=lambda_u,
            use_rel_pos=use_rel_pos,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=ds_dropout_ratio,
        )
        self.uff2 = UFFBlock(
            channels=channels,
            num_classes=num_classes,
            window_size=window_size,
            num_heads=num_heads,
            lambda_u=lambda_u,
            use_rel_pos=use_rel_pos,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=ds_dropout_ratio,
        )
        self.uff1 = UFFBlock(
            channels=channels,
            num_classes=num_classes,
            window_size=window_size,
            num_heads=num_heads,
            lambda_u=lambda_u,
            use_rel_pos=use_rel_pos,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dropout_ratio=ds_dropout_ratio,
        )
        self.conv_seg = None

    def _forward_all(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feats = self._transform_inputs(inputs)
        assert isinstance(feats, (list, tuple)) and len(feats) == 4, 'PSDRHead expects 4 feature maps.'
        f1, f2, f3, f4 = feats

        f1 = self.proj_s1(f1)
        f2 = self.proj_s2(f2)
        f3 = self.proj_s3(f3)
        f4 = self.proj_s4(f4)

        # D4 coarse
        s4 = self.coarse_pred(self.coarse_conv(f4))

        # D3/D2/D1 refinement (residual logits)
        s3 = self.uff3(f3, s4)
        s2 = self.uff2(f2, s3)
        s1 = self.uff1(f1, s2)

        return {'s1': s1, 's2': s2, 's3': s3, 's4': s4}

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        outs = self._forward_all(inputs)
        return outs['s1']

    def loss(self, inputs: List[torch.Tensor], batch_data_samples, train_cfg) -> Dict[str, torch.Tensor]:
        outs = self._forward_all(inputs)

        # main loss on s1
        losses = self.loss_by_feat(outs['s1'], batch_data_samples)

        # deep supervision
        if self.ds_cfg.enable:
            for k in self.ds_cfg.keys:
                if k not in outs:
                    continue
                w = float(self.ds_cfg.weights.get(k, 1.0))
                aux_losses = self.loss_by_feat(outs[k], batch_data_samples)
                for name, val in aux_losses.items():
                    if 'acc' in name:
                        continue
                    losses[f'{name}_{k}'] = val * w

        return losses
