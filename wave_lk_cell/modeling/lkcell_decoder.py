"""LKCell-compatible decoder — matches RepLKDecoder from LKCell exactly.

The only difference: skip_connect=3 always (since our encoder produces 4 feature levels
from 3 encoder stages + 1 wavelet bottleneck stage), and the decoder has 4 UpCat blocks
with nearest-neighbor upsampling, TwoConv (3x3 conv + BN + ReLU) after cat, and
[skip, upsampled] concat order — matching LKCell's MONAI-based decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Sequential:
    """Conv2d(kernel_size, bias=False) + BatchNorm2d."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class TwoConv(nn.Sequential):
    """Two consecutive (Conv3x3 + BN + ReLU) blocks — matches MONAI TwoConv with act=relu."""

    def __init__(self, in_chns: int, out_chns: int) -> None:
        super().__init__()
        self.add_module("conv_0", nn.Sequential(_conv_bn(in_chns, out_chns), nn.ReLU(inplace=True)))
        self.add_module("conv_1", nn.Sequential(_conv_bn(out_chns, out_chns), nn.ReLU(inplace=True)))


class UpCat(nn.Module):
    """Nearest-neighbor upsample x2 → cat([skip, upsampled]) → TwoConv.

    With pre_conv=None (nontrainable upsample), channel count is NOT halved
    during upsampling — matching MONAI UpCat behavior with upsample='nontrainable'.
    """

    def __init__(
        self,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # When upsample='nontrainable' and pre_conv=None, up_chns = in_chns
        # (halves is ignored, matching MONAI's UpCat)
        self.convs = TwoConv(cat_chns + in_chns, out_chns)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor | None = None) -> torch.Tensor:
        x_0 = self.upsample(x)
        if x_e is not None:
            # Pad if spatial dims don't match (from odd encoder sizes)
            if x_0.shape[-2:] != x_e.shape[-2:]:
                diff_h = x_e.shape[-2] - x_0.shape[-2]
                diff_w = x_e.shape[-1] - x_0.shape[-1]
                x_0 = nn.functional.pad(x_0, [0, diff_w, 0, diff_h], mode="replicate")
            x = self.convs(torch.cat([x_e, x_0], dim=1))
        else:
            x = self.convs(x_0)
        return x


class RepLKDecoder(nn.Module):
    """Exact match of LKCell's RepLKDeocder.

    4 UpCat blocks for skip connections + input feature levels.
    RepLKBlock and ConvFFN are instantiated but NOT used in forward (matching LKCell dead code).
    """

    def __init__(
        self,
        encoder_channels: tuple[int, ...],
        decoder_channels: tuple[int, ...] = (1024, 512, 256, 128, 64),
    ) -> None:
        super().__init__()
        # encoder_channels: (3, enc_dim0, enc_dim1, enc_dim2, wavelet_dim)
        # e.g. (3, 96, 192, 384, 768)
        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        # skip_channels: reversed encoder stages (excluding input and last), + [0]
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1) + [False]

        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(UpCat(in_chns=in_chn, cat_chns=skip_chn, out_chns=out_chn))
        self.blocks = nn.ModuleList(blocks)

        # Input feature upsampling levels (matching LKCell lines 154-175)
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

        # Input feature skip convs — TwoConv matching LKCell (lines 174-175)
        # After block[2], x has in_channels[3] = decoder_channels[1] channels
        # input_feature[1] = dims[0]//2, input_feature[0] = dims[0]//4
        input_dim1 = encoder_channels[1] // 2
        input_dim0 = encoder_channels[1] // 4
        self.convs = TwoConv(in_channels[3] + input_dim1, decoder_channels[-2])
        self.convs1 = TwoConv(decoder_channels[-2] + input_dim0, decoder_channels[-1])

    def forward(
        self,
        features: list[torch.Tensor],
        input_feature: list[torch.Tensor],
        skip_connect: int = 3,
    ) -> torch.Tensor:
        skips = features[:-1][::-1]
        x = features[-1]

        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
                x = block(x, skip)
            else:
                # Input feature path — matching LKCell lines 192-200
                skip = input_feature[1]
                x = self.upsample1(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs(x)

                skip = input_feature[0]
                x = self.upsample2(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs1(x)

        return x
