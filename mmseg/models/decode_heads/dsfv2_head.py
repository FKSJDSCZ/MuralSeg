# DSFv2-Decoder head for MuralSeg.
# Proposed: Correlation-guided Confidence-weighted Scale Fusion + Boundary-aware Dual-path Refinement.

import math
import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils import resize


class _DWConvMeanGate(nn.Module):
    """DWConv -> channel-mean -> sigmoid => (B,1,H,W).

    This is the same gating spirit as DSFHead: lightweight spatial gate.
    """

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
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        m = self.dwconv(x).mean(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.sigmoid(m)


@MODELS.register_module()
class DSFv2Head(BaseDecodeHead):
    """DSFv2 Decoder.

    Compared with DSFHead:
      1) Replace independent per-scale gates with *correlation-guided* pixel-wise scale attention.
      2) Predict boundary from (F1, F4) to inject both detail & semantics.
      3) Use boundary-aware dual-path refinement: multi-dilation edge residual + interior smoothing.

    Args:
        interpolate_mode (str): upsample mode. Default: 'bilinear'.
        gate_kernel_size (int): DWConv kernel for self-confidence gate. Default: 3.
        corr_ratio (int): channel reduction ratio for correlation attention. d = channels // corr_ratio.
            Default: 4.
        corr_self_weight_init (float): init value for lambda (self-confidence log term weight).
            Default: 1.0.
        boundary_kernel_size (int): kernel for boundary prediction conv. Default: 3.
        edge_dilations (Sequence[int]): dilation list for edge branches. Default: (1,2,3).
        smooth_kernel_size (int): kernel size for interior smoothing depthwise conv. Default: 5.
        gamma_edge_init (float): init for learnable gamma_edge. Default: 1.0.
        gamma_smooth_init (float): init for learnable gamma_smooth. Default: 0.5.
        eps (float): numerical stability. Default: 1e-6.
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        gate_kernel_size: int = 3,
        corr_ratio: int = 4,
        corr_self_weight_init: float = 1.0,
        boundary_kernel_size: int = 3,
        edge_dilations=(1, 2, 3),
        smooth_kernel_size: int = 5,
        gamma_edge_init: float = 1.0,
        gamma_smooth_init: float = 0.5,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.eps = float(eps)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        assert num_inputs >= 2

        # 1) per-stage 1x1 projection
        self.proj_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                for i in range(num_inputs)
            ],
        )

        # 2) self-confidence gates (per-scale)
        self.self_gates = nn.ModuleList(
            [
                _DWConvMeanGate(self.channels, kernel_size=gate_kernel_size)
                for _ in range(num_inputs)
            ],
        )

        # 3) correlation-guided scale attention (per-pixel, across scales)
        corr_ratio = int(corr_ratio)
        corr_ratio = max(corr_ratio, 1)
        corr_channels = max(self.channels // corr_ratio, 8)
        self.corr_channels = int(corr_channels)

        self.q_proj = Conv2d(self.channels, self.corr_channels, kernel_size=1, bias=True)
        self.k_projs = nn.ModuleList(
            [
                Conv2d(self.channels, self.corr_channels, kernel_size=1, bias=True)
                for _ in range(num_inputs)
            ],
        )

        self.corr_self_weight = nn.Parameter(torch.tensor(float(corr_self_weight_init)))

        self.fuse_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # 4) detail-context coupled boundary prediction
        k = int(boundary_kernel_size)
        assert k % 2 == 1, 'boundary_kernel_size should be odd.'

        self.boundary_in = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.boundary_conv = Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            bias=True,
        )

        # 5) boundary-aware dual-path refinement
        self.edge_dilations = [int(d) for d in edge_dilations]
        self.edge_dilations = [d for d in self.edge_dilations if d >= 1]
        if len(self.edge_dilations) == 0:
            self.edge_dilations = [1]

        self.edge_dwconvs = nn.ModuleList(
            [
                Conv2d(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=d,
                    dilation=d,
                    groups=self.channels,
                    bias=True,
                )
                for d in self.edge_dilations
            ],
        )

        self.edge_weight = Conv2d(
            in_channels=1,
            out_channels=len(self.edge_dilations),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        sk = int(smooth_kernel_size)
        assert sk % 2 == 1, 'smooth_kernel_size should be odd.'
        self.smooth_dwconv = Conv2d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=sk,
            stride=1,
            padding=sk // 2,
            groups=self.channels,
            bias=True,
        )

        self.gamma_edge = nn.Parameter(torch.tensor(float(gamma_edge_init)))
        self.gamma_smooth = nn.Parameter(torch.tensor(float(gamma_smooth_init)))

        self.refine_out = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def forward(self, inputs):
        # inputs: list of multi-level features
        inputs = self._transform_inputs(inputs)
        assert isinstance(inputs, (list, tuple))
        num_inputs = len(inputs)

        tgt_size = inputs[0].shape[2:]

        feats = []
        for idx, x in enumerate(inputs):
            x = self.proj_convs[idx](x)
            x = resize(
                input=x,
                size=tgt_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            feats.append(x)

        # ----- (A) correlation-guided confidence-weighted scale fusion -----
        # self-confidence gate per scale
        g = [gate(feat) for gate, feat in zip(self.self_gates, feats)]  # list of (B,1,H,W)

        # query from averaged multi-scale feature
        f_sum = feats[0]
        for i in range(1, num_inputs):
            f_sum = f_sum + feats[i]
        f_sum = f_sum / float(num_inputs)
        q = self.q_proj(f_sum)  # (B,d,H,W)

        # per-scale score: correlation(q, k_i) + lambda * log(g_i)
        score_list = []
        inv_sqrt_d = 1.0 / math.sqrt(float(self.corr_channels))
        for i in range(num_inputs):
            k = self.k_projs[i](feats[i])
            corr = (q * k).sum(dim=1, keepdim=True) * inv_sqrt_d  # (B,1,H,W)
            conf = torch.log(g[i] + self.eps)
            score = corr + self.corr_self_weight * conf
            score_list.append(score)

        scores = torch.cat(score_list, dim=1)  # (B,N,H,W)
        weights = torch.softmax(scores, dim=1)  # (B,N,H,W)

        fused = 0.0
        for i in range(num_inputs):
            w_i = weights[:, i:i + 1]
            fused = fused + feats[i] * w_i

        fused = self.fuse_conv(fused)

        # ----- (B) detail-context coupled boundary prediction -----
        # use (F1, F4) as (detail, context)
        b_feat = self.boundary_in(torch.cat([feats[0], feats[-1]], dim=1))
        b = torch.sigmoid(self.boundary_conv(b_feat))  # (B,1,H,W)

        # ----- (C) boundary-aware dual-path refinement -----
        # edge residual (multi-dilation) with per-pixel dilation weights conditioned on boundary
        eta = torch.softmax(self.edge_weight(b), dim=1)  # (B,K,H,W)
        edge_res = 0.0
        for j, dw in enumerate(self.edge_dwconvs):
            e = dw(fused)
            edge_res = edge_res + e * eta[:, j:j + 1]

        # interior smoothing
        smooth_res = self.smooth_dwconv(fused)

        fused = fused + self.gamma_edge * (b * edge_res) + self.gamma_smooth * ((1.0 - b) * smooth_res)
        fused = self.refine_out(fused)

        out = self.cls_seg(fused)
        return out
