# DSF-Decoder head for MuralSeg.
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize


class _DWConvMeanGate(nn.Module):
    """DWConv -> channel-mean -> sigmoid => (B,1,H,W)."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        k = int(kernel_size)
        assert k % 2 == 1, 'kernel_size should be odd.'
        self.dwconv = Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=channels,
            bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        m = self.dwconv(x).mean(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.sigmoid(m)


@MODELS.register_module()
class DSFHead(BaseDecodeHead):
    """Detail-Selective Fusion (DSF) Decoder.

    Args:
        interpolate_mode (str): upsample mode, default 'bilinear'
        use_scale_attention (bool): if False, fallback to concat+fusion like official SegformerHead
        use_boundary_refine (bool): enable boundary residual injection
        boundary_gamma_init (float): init value for learnable gamma
        gate_kernel_size (int): DWConv kernel for scale-gates
        boundary_kernel_size (int): Conv kernel for boundary prediction
        eps (float): numerical stability for weight normalization
    """

    def __init__(self,
                 interpolate_mode: str = 'bilinear',
                 use_scale_attention: bool = True,
                 use_boundary_refine: bool = True,
                 boundary_gamma_init: float = 1.0,
                 gate_kernel_size: int = 3,
                 boundary_kernel_size: int = 3,
                 eps: float = 1e-6,
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.use_scale_attention = bool(use_scale_attention)
        self.use_boundary_refine = bool(use_boundary_refine)
        self.eps = float(eps)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        # 1) per-stage 1x1 projection (same spirit as official SegformerHead)
        self.proj_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.proj_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # 2) fusion
        if self.use_scale_attention:
            self.scale_gates = nn.ModuleList([
                _DWConvMeanGate(self.channels, kernel_size=gate_kernel_size)
                for _ in range(num_inputs)
            ])
            # lightweight post-fusion mixing
            self.fuse_conv = ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            # fallback to official style: concat + 1x1 fusion
            self.fuse_conv = ConvModule(
                in_channels=self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        # 3) boundary residual injection
        if self.use_boundary_refine:
            k = int(boundary_kernel_size)
            assert k % 2 == 1, 'boundary_kernel_size should be odd.'
            self.boundary_conv = Conv2d(
                in_channels=self.channels,
                out_channels=1,
                kernel_size=k,
                stride=1,
                padding=k // 2,
                bias=True)

            self.refine_dwconv = Conv2d(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.channels,
                bias=True)

            self.boundary_gamma = nn.Parameter(torch.tensor(float(boundary_gamma_init)))

            self.refine_out = ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def forward(self, inputs):
        # inputs: list of multi-level features
        inputs = self._transform_inputs(inputs)
        assert isinstance(inputs, (list, tuple)) and len(inputs) >= 2

        # target size: highest resolution among selected inputs (usually stage0: 1/4)
        tgt_size = inputs[0].shape[2:]

        feats = []
        for idx, x in enumerate(inputs):
            x = self.proj_convs[idx](x)
            x = resize(
                input=x,
                size=tgt_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners)
            feats.append(x)

        if self.use_scale_attention:
            gates = [g(feat) for g, feat in zip(self.scale_gates, feats)]  # each (B,1,H,W)
            denom = gates[0]
            for i in range(1, len(gates)):
                denom = denom + gates[i]
            denom = denom + self.eps

            fused = 0.0
            for feat, gate in zip(feats, gates):
                fused = fused + feat * (gate / denom)  # pixel-wise weighted sum
            fused = self.fuse_conv(fused)
        else:
            fused = self.fuse_conv(torch.cat(feats, dim=1))

        if self.use_boundary_refine:
            # boundary from highest-res projected feature
            b = torch.sigmoid(self.boundary_conv(feats[0]))  # (B,1,H,W)
            refined = self.refine_dwconv(fused)  # (B,C,H,W)
            fused = fused + self.boundary_gamma * (b * refined)
            fused = self.refine_out(fused)

        out = self.cls_seg(fused)
        return out
