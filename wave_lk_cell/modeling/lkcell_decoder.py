from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        small_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.lkb_origin = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        if small_kernel is not None:
            assert small_kernel <= kernel_size
            self.small_conv = nn.Sequential(
                nn.Conv2d(channels, channels, small_kernel, stride=1, padding=small_kernel // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(x)
        return out


class ConvFFN(nn.Module):
    def __init__(self, in_channels: int, internal_channels: int, out_channels: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, 1, bias=False),
            nn.BatchNorm2d(internal_channels),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.nonlinear = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class LKCellDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        large_kernel_size: int = 13,
        small_kernel: int = 5,
        drop_path: float = 0.0,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.lk_block = ReparamLargeKernelConv(in_channels, large_kernel_size, small_kernel)
        self.lk_bn = nn.BatchNorm2d(in_channels)
        self.lk_act = nn.ReLU(inplace=True)
        self.lk_pw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        cat_channels = in_channels + skip_channels
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.ffn = ConvFFN(out_channels, out_channels * ffn_ratio, out_channels, drop_path)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.lk_bn(x)
        out = self.lk_block(out)
        out = self.lk_act(out)
        out = self.lk_pw(out)
        out = out + x

        out = self.upsample(out)
        out = torch.cat([out, skip], dim=1)
        out = self.cat_conv(out)
        out = self.ffn(out)
        return out


class LKCellDecoder(nn.Module):
    def __init__(
        self,
        encoder_dims: tuple[int, ...],
        decoder_channels: tuple[int, ...],
        large_kernel_sizes: tuple[int, ...] = (13, 27, 29),
        small_kernel: int = 5,
        drop_path: float = 0.1,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        assert len(encoder_dims) == 4
        assert len(decoder_channels) == 5
        assert len(large_kernel_sizes) == 3

        in_channels_list = [encoder_dims[-1]] + list(decoder_channels[:3])
        skip_channels_list = list(encoder_dims[:3][::-1])

        self.blocks = nn.ModuleList()
        for i in range(3):
            self.blocks.append(
                LKCellDecoderBlock(
                    in_channels=in_channels_list[i],
                    skip_channels=skip_channels_list[i],
                    out_channels=decoder_channels[i],
                    large_kernel_size=large_kernel_sizes[i],
                    small_kernel=small_kernel,
                    drop_path=drop_path,
                    ffn_ratio=ffn_ratio,
                )
            )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.input_skip_conv1 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + encoder_dims[0] // 2, decoder_channels[3], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[3], decoder_channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )
        self.input_skip_conv2 = nn.Sequential(
            nn.Conv2d(decoder_channels[3] + encoder_dims[0] // 4, decoder_channels[4], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[4], decoder_channels[4], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[4]),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        stage_features: list[torch.Tensor],
        input_features: list[torch.Tensor],
    ) -> torch.Tensor:
        skips = stage_features[:3][::-1]
        x = stage_features[3]

        for i, block in enumerate(self.blocks):
            x = block(x, skips[i])

        x = self.upsample1(x)
        x = torch.cat([x, input_features[1]], dim=1)
        x = self.input_skip_conv1(x)

        x = self.upsample2(x)
        x = torch.cat([x, input_features[0]], dim=1)
        x = self.input_skip_conv2(x)

        return x
