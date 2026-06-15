# DSFv2-Decoder head for MuralSeg.
# Proposed: Correlation-guided Confidence-weighted Scale Fusion (C2SF)
#           + Detail-Context Coupled Boundary (merged into BDPR)
#           + Boundary-aware Dual-path Refinement (BDPR)
#
# This implementation is ablation-friendly:
#   (1) C2SF / BDPR on-off ablations.
#   (2) Alternatives to the log-combination in C2SF.

import math
from typing import List, Optional

import torch
import torch.nn as nn
from mmcv.cnn import Conv2d, ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS


def _safe_logit(p: torch.Tensor, eps: float) -> torch.Tensor:
    # logit(p) = log(p/(1-p)) with numerical safety.
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


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


class _C2SF(nn.Module):
    """
    Correlation-guided Confidence-weighted Scale Fusion (C2SF).

    Given aligned multi-scale features {F_i}, compute per-pixel scale weights
    and return fused feature.

    score_type controls how relevance(corr) and confidence(gate) are combined:
        - 'add'       : corr + lambda * conf
        - 'corr_only' : corr only (no confidence term)
        - 'conv'      : per-scale 1x1 conv on [corr, conf] -> score
        - 'mlp'       : per-scale 2-layer 1x1 conv MLP on [corr, conf] -> score

    conf_transform is applied to g before combining (for conv/mlp it's the
    second input channel):
        - 'log'    : conf = log(g + eps)
        - 'linear' : conf = g
        - 'logit'  : conf = logit(g, eps)
    """

    def __init__(
        self,
        channels: int,
        num_scales: int,
        gate_kernel_size: int = 3,
        corr_ratio: int = 4,
        corr_self_weight_init: float = 1.0,
        score_type: str = 'add',
        conf_transform: str = 'log',
        learned_fuser_hidden: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_scales = int(num_scales)
        self.eps = float(eps)

        self.score_type = str(score_type)
        self.conf_transform = str(conf_transform)

        # (a) self-confidence gates
        self.self_gates = nn.ModuleList(
            [_DWConvMeanGate(self.channels, kernel_size=gate_kernel_size) for _ in range(self.num_scales)],
        )

        # (b) correlation attention (Q from mean(F_i), K_i from each scale)
        corr_ratio = max(int(corr_ratio), 1)
        corr_channels = max(self.channels // corr_ratio, 8)
        self.corr_channels = int(corr_channels)

        self.q_proj = Conv2d(self.channels, self.corr_channels, kernel_size=1, bias=True)
        self.k_projs = nn.ModuleList(
            [Conv2d(self.channels, self.corr_channels, kernel_size=1, bias=True) for _ in range(self.num_scales)],
        )

        # lambda in: corr + lambda * conf_term
        if self.score_type == 'add':
            self.corr_self_weight = nn.Parameter(torch.tensor(float(corr_self_weight_init)))
        else:
            self.corr_self_weight = None

        # learned fusers for conv/mlp variants
        self.score_fusers: Optional[nn.ModuleList]
        if self.score_type in {'conv', 'mlp'}:
            hidden = max(int(learned_fuser_hidden), 4)
            if self.score_type == 'conv':
                self.score_fusers = nn.ModuleList(
                    [Conv2d(2, 1, kernel_size=1, bias=True) for _ in range(self.num_scales)],
                )
            else:
                self.score_fusers = nn.ModuleList(
                    [
                        nn.Sequential(
                            Conv2d(2, hidden, kernel_size=1, bias=True),
                            nn.ReLU(inplace=True),
                            Conv2d(hidden, 1, kernel_size=1, bias=True),
                        )
                        for _ in range(self.num_scales)
                    ],
                )
        else:
            self.score_fusers = None

    def _apply_conf_transform(self, g: torch.Tensor) -> torch.Tensor:
        if self.conf_transform == 'linear':
            return g
        if self.conf_transform == 'log':
            return torch.log(g + self.eps)
        if self.conf_transform == 'logit':
            return _safe_logit(g, eps=self.eps)
        raise ValueError(
            f"Unknown conf_transform={self.conf_transform}. Use one of {'log', 'linear', 'logit'}.",
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(feats, (list, tuple))
        assert len(feats) == self.num_scales

        # self-confidence gate per scale (B,1,H,W)
        g = [gate(feat) for gate, feat in zip(self.self_gates, feats)]

        # query from averaged multi-scale feature
        f_sum = feats[0]
        for i in range(1, self.num_scales):
            f_sum = f_sum + feats[i]
        f_sum = f_sum / float(self.num_scales)
        q = self.q_proj(f_sum)  # (B,d,H,W)

        inv_sqrt_d = 1.0 / math.sqrt(float(self.corr_channels))

        score_list = []
        for i in range(self.num_scales):
            k = self.k_projs[i](feats[i])
            corr = (q * k).sum(dim=1, keepdim=True) * inv_sqrt_d  # (B,1,H,W)
            conf = self._apply_conf_transform(g[i])

            if self.score_type == 'add':
                score = corr + self.corr_self_weight * conf
            elif self.score_type == 'corr_only':
                score = corr
            elif self.score_type in {'conv', 'mlp'}:
                score = self.score_fusers[i](torch.cat([corr, conf], dim=1))
            else:
                raise ValueError(
                    f"Unknown score_type={self.score_type}. Use one of {'add', 'corr_only', 'conv', 'mlp'}.",
                )

            score_list.append(score)

        scores = torch.cat(score_list, dim=1)  # (B,N,H,W)
        weights = torch.softmax(scores, dim=1)  # (B,N,H,W)

        fused = 0.0
        for i in range(self.num_scales):
            fused = fused + feats[i] * weights[:, i:i + 1]
        return fused


class _BDPR(nn.Module):
    """BDPR = (DCC-Boundary + Dual-path Refinement).

    In this codebase, we treat DCC-Boundary as part of BDPR for ablation:
      - enable_bdpr=False => both boundary prediction and refinement are removed.

    Args:
        channels (int): feature channels.
        norm_cfg/act_cfg: passed into ConvModule.
    """

    def __init__(
        self,
        channels: int,
        *,
        norm_cfg,
        act_cfg,
        boundary_kernel_size: int = 3,
        edge_dilations=(1, 2, 3),
        smooth_kernel_size: int = 5,
        gamma_edge_init: float = 1.0,
        gamma_smooth_init: float = 0.5,
    ):
        super().__init__()
        self.channels = int(channels)

        # ----- DCC-Boundary -----
        k = int(boundary_kernel_size)
        assert k % 2 == 1, 'boundary_kernel_size should be odd.'

        self.boundary_in = ConvModule(
            in_channels=self.channels * 2,
            out_channels=self.channels,
            kernel_size=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.boundary_conv = Conv2d(
            in_channels=self.channels,
            out_channels=1,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            bias=True,
        )

        # ----- BDPR refinement -----
        edge_dilations = [int(d) for d in edge_dilations]
        edge_dilations = [d for d in edge_dilations if d >= 1]
        if len(edge_dilations) == 0:
            edge_dilations = [1]
        self.edge_dilations = edge_dilations

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
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, fused: torch.Tensor, feat_detail: torch.Tensor, feat_context: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W)
        b_feat = self.boundary_in(torch.cat([feat_detail, feat_context], dim=1))
        b = torch.sigmoid(self.boundary_conv(b_feat))  # (B,1,H,W)

        # edge residual (multi-dilation) with per-pixel dilation weights conditioned on boundary
        eta = torch.softmax(self.edge_weight(b), dim=1)  # (B,K,H,W)
        edge_res = 0.0
        for j, dw in enumerate(self.edge_dwconvs):
            edge_res = edge_res + dw(fused) * eta[:, j:j + 1]

        # interior smoothing
        smooth_res = self.smooth_dwconv(fused)

        fused = fused + self.gamma_edge * (b * edge_res) + self.gamma_smooth * ((1.0 - b) * smooth_res)
        fused = self.refine_out(fused)
        return fused


@MODELS.register_module()
class DSFv2HeadAblation(BaseDecodeHead):
    """DSFv2 Decoder (ablation-friendly).

    Modules:
      - C2SF: correlation-guided confidence-weighted scale fusion.
      - BDPR: (DCC-Boundary + dual-path refinement).

    Ablation knobs:
      - enable_c2sf: turn off => use `c2sf_fallback` fusion.
      - enable_bdpr: turn off => remove boundary prediction + refinement.
      - c2sf_score_type: alternative ways to combine corr/conf in C2SF.

    Note: for paper ablations where DCC-Boundary is considered part of BDPR,
          set enable_bdpr=False to remove both.
    """

    def __init__(
        self,
        interpolate_mode: str = 'bilinear',
        # C2SF
        enable_c2sf: bool = True,
        gate_kernel_size: int = 3,
        corr_ratio: int = 4,
        corr_self_weight_init: float = 1.0,
        c2sf_score_type: str = 'add',
        c2sf_conf_transform: str = 'log',
        c2sf_learned_fuser_hidden: int = 8,
        c2sf_fallback: str = 'concat',
        # BDPR (includes DCC-Boundary)
        enable_bdpr: bool = True,
        boundary_kernel_size: int = 3,
        edge_dilations=(1, 2, 3),
        smooth_kernel_size: int = 5,
        gamma_edge_init: float = 1.0,
        gamma_smooth_init: float = 0.5,
        # misc
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = str(interpolate_mode)
        self.eps = float(eps)

        self.enable_c2sf = bool(enable_c2sf)
        self.enable_bdpr = bool(enable_bdpr)
        self.c2sf_fallback = str(c2sf_fallback)

        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        assert num_inputs >= 2
        self.num_inputs = int(num_inputs)

        # 1) per-stage 1x1 projection to `self.channels`
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
                for i in range(self.num_inputs)
            ],
        )

        # If C2SF is disabled and fallback is concat, we need a reduction conv.
        self._naive_concat_conv: Optional[nn.Module] = None
        if (not self.enable_c2sf) and self.c2sf_fallback == 'concat':
            self._naive_concat_conv = ConvModule(
                in_channels=self.channels * self.num_inputs,
                out_channels=self.channels,
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )

        # 2) C2SF fusion module (optional)
        self.c2sf: Optional[_C2SF] = None
        if self.enable_c2sf:
            self.c2sf = _C2SF(
                channels=self.channels,
                num_scales=self.num_inputs,
                gate_kernel_size=gate_kernel_size,
                corr_ratio=corr_ratio,
                corr_self_weight_init=corr_self_weight_init,
                score_type=c2sf_score_type,
                conf_transform=c2sf_conf_transform,
                learned_fuser_hidden=c2sf_learned_fuser_hidden,
                eps=self.eps,
            )

        # 3) post-fusion channel mixing
        self.post_fuse_conv = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # 4) BDPR module (optional, includes boundary prediction)
        self.bdpr: Optional[_BDPR] = None
        if self.enable_bdpr:
            self.bdpr = _BDPR(
                channels=self.channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                boundary_kernel_size=boundary_kernel_size,
                edge_dilations=edge_dilations,
                smooth_kernel_size=smooth_kernel_size,
                gamma_edge_init=gamma_edge_init,
                gamma_smooth_init=gamma_smooth_init,
            )

    def _project_and_align(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project and resize multi-level features to a common resolution."""
        tgt_size = inputs[0].shape[2:]
        feats: List[torch.Tensor] = []
        for idx, x in enumerate(inputs):
            x = self.proj_convs[idx](x)
            x = resize(
                input=x,
                size=tgt_size,
                mode=self.interpolate_mode,
                align_corners=self.align_corners,
            )
            feats.append(x)
        return feats

    def _naive_fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Fallback fusion when C2SF is disabled."""
        if self.c2sf_fallback in {'avg', 'mean'}:
            fused = feats[0]
            for i in range(1, self.num_inputs):
                fused = fused + feats[i]
            fused = fused / float(self.num_inputs)
            return fused

        if self.c2sf_fallback == 'sum':
            fused = feats[0]
            for i in range(1, self.num_inputs):
                fused = fused + feats[i]
            return fused

        if self.c2sf_fallback == 'concat':
            assert self._naive_concat_conv is not None
            fused = self._naive_concat_conv(torch.cat(feats, dim=1))
            return fused

        raise ValueError(
            f"Unknown c2sf_fallback={self.c2sf_fallback}. Use one of {'avg', 'sum', 'concat'}.",
        )

    def forward(self, inputs):
        # inputs: list of multi-level features
        inputs = self._transform_inputs(inputs)
        assert isinstance(inputs, (list, tuple))

        feats = self._project_and_align(list(inputs))

        # ----- (A) scale fusion -----
        if self.enable_c2sf:
            assert self.c2sf is not None
            fused = self.c2sf(feats)
        else:
            fused = self._naive_fuse(feats)

        fused = self.post_fuse_conv(fused)

        # ----- (B) boundary-aware refinement (DCC-Boundary is inside) -----
        if self.enable_bdpr:
            assert self.bdpr is not None
            fused = self.bdpr(fused, feat_detail=feats[0], feat_context=feats[-1])

        out = self.cls_seg(fused)
        return out
