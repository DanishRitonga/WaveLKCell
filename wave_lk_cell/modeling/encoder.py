# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------
# Adapted from LKCell for WaveLKCell — removed MMSeg/MMDet, iGEMM, deploy mode,
# classification head, gradient checkpointing, and other unnecessary options.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import trunc_normal_, DropPath, to_2tuple


class GRNwithNHWC(nn.Module):
    """GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (
        conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
        bn.bias + (conv_bias - bn.running_mean) * bn.weight / std,
    )


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1))
    if kernel.size(1) == 1:
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(
                kernel[:, i : i + 1, :, :], identity_kernel, stride=dilate_rate
            )
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.lk_origin = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=False,
        )

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError("Dilated Reparam Block requires kernel_size >= 5")

        self.origin_bn = nn.BatchNorm2d(channels)
        for k, r in zip(self.kernel_sizes, self.dilates):
            self.__setattr__(
                "dil_conv_k{}_{}".format(k, r),
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    stride=1,
                    padding=(r * (k - 1) + 1) // 2,
                    dilation=r,
                    groups=channels,
                    bias=False,
                ),
            )
            self.__setattr__(
                "dil_bn_k{}_{}".format(k, r), nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__("dil_conv_k{}_{}".format(k, r))
            bn = self.__getattr__("dil_bn_k{}_{}".format(k, r))
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        ffn_factor=4,
    ):
        super().__init__()

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
                groups=dim,
                bias=False,
            )

        if kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = nn.BatchNorm2d(dim)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(ffn_dim, use_bias=True))
        self.pwconv2 = nn.Sequential(
            nn.Linear(ffn_dim, dim, bias=False),
            NHWCtoNCHW(),
            nn.BatchNorm2d(dim),
        )

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True
            )
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):
        return inputs + self.compute_residual(inputs)


default_UniRepLKNet_A_F_P_kernel_sizes = (
    (3, 3),
    (13, 13),
    (13, 13, 13, 13, 13, 13),
    (13, 13),
)
default_UniRepLKNet_N_kernel_sizes = (
    (3, 3),
    (13, 13),
    (13, 13, 13, 13, 13, 13, 13, 13),
    (13, 13),
)
default_UniRepLKNet_T_kernel_sizes = (
    (3, 3, 3),
    (13, 13, 13),
    (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
    (13, 13, 13),
)
default_UniRepLKNet_S_B_L_XL_kernel_sizes = (
    (3, 3, 3),
    (13, 13, 13),
    (
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
        13,
        3,
        3,
    ),
    (13, 13, 13),
)
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes,
}


class LayerNorm(nn.Module):
    r"""LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_last",
        reshape_last_to_first=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class UniRepLKNet(nn.Module):
    r"""UniRepLKNet encoder (feature extractor only, no classification head).

    Returns:
        (features_list, input_features_list) where:
            features_list = [stage0_out, stage1_out, stage2_out, stage3_out]  (each B,C,H,W)
            input_features_list = [input_conv_out, input_down_conv_out]  (each B,C,H,W)

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 27, 3)
        dims (int): Feature dimension at each stage. Default: (96, 192, 384, 768)
        drop_path_rate (float): Stochastic depth rate. Default: 0.3
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6
        kernel_sizes (tuple(tuple(int))): Kernel size for each block. None means using the default settings.
    """

    def __init__(
        self,
        in_chans=3,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=0.3,
        layer_scale_init_value=1e-6,
        kernel_sizes=None,
        **kwargs,
    ):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError(
                    "no default kernel size settings for the given depths, "
                    "please specify kernel sizes for each block, e.g., "
                    "((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))"
                )
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], (
                "kernel sizes do not match the depths"
            )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1
                ),
                LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )

        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    LayerNorm(
                        dims[i + 1], eps=1e-6, data_format="channels_first"
                    ),
                )
            )

        self.stages = nn.ModuleList()

        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[
                    UniRepLKNetBlock(
                        dim=dims[i],
                        kernel_size=kernel_sizes[i][j],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(main_stage)
            cur += depths[i]

        self.conv = nn.Conv2d(
            in_chans, dims[0] // 4, kernel_size=3, stride=1, padding=1
        )

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input_features = []
        input_features.append(self.conv(x))
        input_features.append(self.downsample_layers[0][0](x))

        features = []
        raw_features = []
        for stage_idx in range(4):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
            raw_features.append(x)
            features.append(self.__getattr__(f"norm{stage_idx}")(x))

        return features, input_features, raw_features
