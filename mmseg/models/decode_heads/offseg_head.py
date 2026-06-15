import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmengine.model.weight_init import constant_init, normal_init, trunc_normal_, trunc_normal_init, xavier_init
# from mmcv.ops.carafe import carafe
from mmcv.cnn import build_norm_layer, ConvModule
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    # assert m_c == kernel_size ** 2 * up ** 2
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2
    # print(pad)
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    # print(pad_x.shape)
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res


def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class FreqFusion(nn.Module):
    def __init__(
        self,
        hr_channels,
        lr_channels,
        scale_factor=1,
        lowpass_kernel=5,
        highpass_kernel=3,
        up_group=1,
        encoder_kernel=3,
        encoder_dilation=1,
        compressed_channels=64,
        align_corners=False,
        upsample_mode='nearest',
        feature_resample=True,  # use offset generator or not
        feature_resample_group=4,
        comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
        use_high_pass=True,
        use_low_pass=True,
        hr_residual=True,
        semi_conv=True,
        hamming_window=False,  # for regularization, do not matter really
        feature_resample_norm=True,
        **kwargs,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(  # ALPF generator
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1,
        )

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(
                in_channels=compressed_channels, scale=2, style='lp', groups=feature_resample_group,
                use_direct_scale=True, kernel_size=encoder_kernel, norm=feature_resample_norm,
            )
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
            )
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer(
                'hamming_lowpass', torch.FloatTensor(
                    hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad),
                )[None, None,],
            )
            self.register_buffer(
                'hamming_highpass', torch.FloatTensor(
                    hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad),
                )[None, None,],
            )
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        # print(hamming)
        # print(mask.shape)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, hr_feat, lr_feat)
        else:
            return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(
                        mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass,
                    )
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(
                        compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1,
                    )

                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(
                        mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass,
                    )

                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                        size=compressed_hr_feat.shape[-2:], mode='nearest',
                    )
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(
                            self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group,
                            2,
                        ), size=compressed_hr_feat.shape[-2:], mode='nearest',
                    )
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else:
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest',
                )
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest',
                    )
        else:
            compressed_x = F.interpolate(
                compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest',
            ) + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == 'nearest' else self.align_corners,
            )
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            if self.hr_residual:
                # print('using hr_residual')
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        if self.feature_resample:
            # print(lr_feat.shape)
            lr_feat = self.dysampler(
                hr_x=compressed_hr_feat,
                lr_x=compressed_lr_feat, feat2sample=lr_feat,
            )

        return mask_lr, hr_feat, lr_feat


class LocalSimGuidedSampler(nn.Module):
    """
    offset generator in FreqFusion
    """

    def __init__(
        self, in_channels, scale=2, style='lp', groups=4, use_direct_scale=True, kernel_size=1, local_window=3,
        sim_type='cos', norm=True, direction_feat='sim',
    ):
        super().__init__()
        assert scale == 2
        assert style == 'lp'

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2
        if self.direction_feat == 'sim':
            self.offset = nn.Conv2d(
                local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
            )
        elif self.direction_feat == 'sim_concat':
            self.offset = nn.Conv2d(
                in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
            )
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)
        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.direct_scale = nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                )
            elif self.direction_feat == 'sim_concat':
                self.direct_scale = nn.Conv2d(
                    in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.)

        out_channels = 2 * groups
        if self.direction_feat == 'sim':
            self.hr_offset = nn.Conv2d(
                local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
            )
        elif self.direction_feat == 'sim_concat':
            self.hr_offset = nn.Conv2d(
                in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
            )
        else:
            raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == 'sim':
                self.hr_direct_scale = nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                )
            elif self.direction_feat == 'sim_concat':
                self.hr_direct_scale = nn.Conv2d(
                    in_channels + local_window ** 2 - 1, out_channels, kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            else:
                raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset, scale=None):
        if scale is None:
            scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(
            torch.meshgrid([coords_w, coords_h]),
        ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(
            B, 2, -1, scale * H, scale * W,
        ).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)), coords, mode='bilinear',
            align_corners=False, padding_mode="border",
        ).view(B, -1, scale * H, scale * W)

    def forward(self, hr_x, lr_x, feat2sample):
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == 'sim':
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')
        elif self.direction_feat == 'sim_concat':
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim='cos')], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        # offset = self.get_offset(hr_x, lr_x)
        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)

    # def get_offset_lp(self, hr_x, lr_x):
    def get_offset_lp(self, hr_x, lr_x, hr_sim, lr_sim):
        if hasattr(self, 'direct_scale'):
            # offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * (self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)).sigmoid() + self.init_pos
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (
                    self.direct_scale(lr_x) + F.pixel_unshuffle(
                self.hr_direct_scale(hr_x), self.scale,
            )).sigmoid() + self.init_pos
            # offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (self.direct_scale(lr_sim) + F.pixel_unshuffle(self.hr_direct_scale(hr_sim), self.scale)).sigmoid() + self.init_pos
        else:
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def get_offset(self, hr_x, lr_x):
        if self.style == 'pl':
            raise NotImplementedError
        return self.get_offset_lp(hr_x, lr_x)


def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    """
    计算输入张量中每一点与周围KxK范围内的点的余弦相似度。

    参数：
    - input_tensor: 输入张量，形状为[B, C, H, W]
    - k: 范围大小，表示周围KxK范围内的点

    返回：
    - 输出张量，形状为[B, KxK-1, H, W]
    """
    B, C, H, W = input_tensor.shape
    # 使用零填充来处理边界情况
    # padded_input = F.pad(input_tensor, (k // 2, k // 2, k // 2, k // 2), mode='constant', value=0)

    # 展平输入张量中每个点及其周围KxK范围内的点
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
    # print(unfold_tensor.shape)
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)

    # 计算余弦相似度
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    # 移除中心点的余弦相似度，得到[KxK-1]的结果
    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)

    # 将结果重塑回[B, KxK-1, H, W]的形状
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


class Offset_Learning(nn.Module):
    """
    Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment

    https://arxiv.org/abs/2508.08811
    """

    def __init__(self, num_classes, embed_dims, init_std=0.02, norm_cfg=dict(type='LN'), ):
        super(Offset_Learning, self).__init__()
        self.num_classes = num_classes
        self.cls_repr = nn.Parameter(
            torch.randn(1, num_classes, embed_dims),
        )
        self.init_std = init_std
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=1,
        )[1]
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.cls_repr, std=self.init_std)
        trunc_normal_init(self.cls_offset_proj, std=self.init_std)
        trunc_normal_init(self.feat_offset_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, x):
        b, c, h, w = x.shape
        cls_repr = self.cls_repr.expand(b, -1, -1)  # b, k, c
        img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # b, hw, c

        # compute coupled attention
        coupled_attn = img_feat @ cls_repr.transpose(1, 2)  # b, hw, k
        coupled_attn = coupled_attn.permute(0, 2, 1)  # b, k, hw

        # class offset learning
        cls_attn = coupled_attn.softmax(dim=2)  # b, k, hw
        cls_offset = self.cls_offset_proj(cls_attn @ img_feat)  # b, k, c
        aligned_cls_repr = cls_repr + cls_offset  # b, k, c

        # feature offset learning
        pos_attn = coupled_attn.softmax(dim=1)  # b, k, hw
        feat_offset = self.feat_offset_proj(pos_attn.transpose(1, 2) @ cls_repr)  # b, hw, c
        aligned_img_feat = img_feat + feat_offset  # b, hw, c

        # compute masks
        masks = aligned_img_feat @ aligned_cls_repr.transpose(1, 2)  # b, hw, k
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        return masks


class Offset_Learning_Mask(nn.Module):
    """
    ICCV 2025: Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>
    """

    def __init__(self, embed_dims, init_std=0.02, norm_cfg=dict(type='LN'), ):
        super(Offset_Learning_Mask, self).__init__()
        self.init_std = init_std
        self.cls_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.feat_offset_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.init_weights()

    def init_weights(self):
        trunc_normal_init(self.cls_offset_proj, std=self.init_std)
        trunc_normal_init(self.feat_offset_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, mask_embed, mask_features):
        b, c, h, w = mask_features.shape
        cls_repr = mask_embed  # b, k, c
        img_feat = mask_features.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)  # b, hw, c

        # compute coupled attention
        coupled_attn = img_feat @ cls_repr.transpose(1, 2)  # b, hw, k
        coupled_attn = coupled_attn.permute(0, 2, 1)  # b, k, hw

        # class offset learning
        cls_attn = coupled_attn.softmax(dim=2)
        cls_offset = self.cls_offset_proj(cls_attn @ img_feat)
        aligned_cls_repr = cls_repr + cls_offset

        # feature offset learning
        pos_attn = coupled_attn.softmax(dim=1)
        feat_offset = self.feat_offset_proj(pos_attn.transpose(1, 2) @ cls_repr)
        aligned_img_feat = img_feat + feat_offset
        aligned_img_feat = aligned_img_feat.transpose(1, 2).contiguous().view(b, c, h, w)

        return aligned_cls_repr, aligned_img_feat


@MODELS.register_module()
class OffSegHead(BaseDecodeHead):
    """
    OffSeg decode head.

    This decode head is the implementation of `Revisiting Efficient Semantic Segmentation: Learning Offsets for Better Spatial and Class Feature Alignment
    <https://arxiv.org/abs/2508.08811>`_.

    Args:
        in_channels (list): input channels for OffSeg.
        new_channels (list): hidden channels for OffSeg.
        num_classes (int): number of classes.
    """

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            input_transform='multiple_select',
            **kwargs,
        )
        self.conv_seg.requires_grad_(False)  # find_unused_parameters
        self.new_channels = new_channels

        self.pre = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pre.append(
                ConvModule(
                    self.in_channels[i],
                    self.new_channels[i],
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            )

        self.freqfusions = nn.ModuleList()
        in_channels = new_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c, lr_channels=pre_c,
                compressed_channels=(pre_c + c) // 4,
            )
            self.freqfusions.append(freqfusion)
            pre_c += c

        self.align = ConvModule(
            sum(self.new_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        # delattr(self, 'conv_seg')
        self.offset_learning = Offset_Learning(self.num_classes, self.channels)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        new_inputs = []
        for i in range(len(inputs)):
            new_inputs.append(self.pre[i](inputs[i]))

        inputs = new_inputs

        inputs = inputs[::-1]
        lowres_feat = inputs[0]
        for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [
                    hires_feat.reshape(b * 4, -1, h, w),
                    lowres_feat.reshape(b * 4, -1, h, w),
                ], dim=1,
            ).reshape(b, -1, h, w)

        inputs = lowres_feat

        output = self.align(inputs)
        output = self.offset_learning(output)
        return output
