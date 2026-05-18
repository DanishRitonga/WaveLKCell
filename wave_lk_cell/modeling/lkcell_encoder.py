from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class GRNwithNHWC(nn.Module):
    def __init__(self, dim: int, use_bias: bool = True) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    def __init__(self, input_channels: int, internal_neurons: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(inputs, 1)
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DilatedReparamBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, deploy: bool = False) -> None:
        super().__init__()
        self.lk_origin = nn.Conv2d(
            channels, channels, kernel_size, stride=1,
            padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
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
            raise ValueError(f"DilatedReparamBlock requires kernel_size >= 5, got {kernel_size}")

        if not deploy:
            self.origin_bn = nn.BatchNorm2d(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(
                    f"dil_conv_k{k}_{r}",
                    nn.Conv2d(
                        channels, channels, k, stride=1,
                        padding=(r * (k - 1) + 1) // 2, dilation=r,
                        groups=channels, bias=False,
                    ),
                )
                self.__setattr__(f"dil_bn_k{k}_{r}", nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "origin_bn"):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f"dil_conv_k{k}_{r}")
            bn = self.__getattr__(f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out


class UniRepLKNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        deploy: bool = False,
        ffn_factor: int = 4,
    ) -> None:
        super().__init__()

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy)
        else:
            assert kernel_size in [3, 5]
            self.dwconv = nn.Conv2d(
                dim, dim, kernel_size, stride=1, padding=kernel_size // 2,
                groups=dim, bias=deploy,
            )

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = nn.BatchNorm2d(dim)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim),
        )
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy),
        )
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW(),
            )
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                nn.BatchNorm2d(dim),
            )

        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if (not deploy) and layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        y = self.se(self.norm(self.dwconv(inputs)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return inputs + self.drop_path(y)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm = LayerNormChannelsFirst(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class LKCellEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depths: tuple[int, ...] = (3, 3, 27),
        dims: tuple[int, ...] = (96, 192, 384),
        drop_path_rate: float = 0.3,
        layer_scale_init_value: float = 1e-6,
        kernel_sizes: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_stages = len(depths)
        self.dims = dims

        default_kernel_sizes = {
            (2, 2, 6): ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13)),
            (2, 2, 8): ((3, 3), (13, 13), (13, 13, 13, 13, 13, 13, 13, 13)),
            (3, 3, 27): (
                (3, 3, 3), (13, 13, 13),
                (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
            ),
        }
        if kernel_sizes is None:
            if depths in default_kernel_sizes:
                kernel_sizes = default_kernel_sizes[depths]
            else:
                raise ValueError(
                    f"No default kernel sizes for depths={depths}. "
                    "Please provide kernel_sizes explicitly."
                )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, 3, stride=2, padding=1),
            LayerNormChannelsFirst(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1),
            LayerNormChannelsFirst(dims[0]),
        ))
        for i in range(len(depths) - 1):
            self.downsample_layers.append(
                DownsampleLayer(dims[i], dims[i + 1], stride=2)
            )

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
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
            self.stages.append(stage)
            cur += depths[i]

        self.norms = nn.ModuleList([
            LayerNormChannelsFirst(dims[i]) for i in range(len(depths))
        ])

        self.input_conv = nn.Conv2d(in_channels, dims[0] // 4, 3, stride=1, padding=1)
        self.input_down_conv = nn.Conv2d(in_channels, dims[0] // 2, 3, stride=2, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        input_features = [self.input_conv(x), self.input_down_conv(x)]
        stage_features = []
        for stage_idx in range(self.num_stages):
            x = self.downsample_layers[stage_idx](x)
            x = self.stages[stage_idx](x)
            x = self.norms[stage_idx](x)
            stage_features.append(x)
        return stage_features, input_features
