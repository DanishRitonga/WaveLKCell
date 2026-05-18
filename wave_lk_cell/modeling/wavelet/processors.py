from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePowerGaborConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        half = kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32),
            torch.arange(-half, half + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_y", y)
        self.register_buffer("grid_x", x)

        self.wavelength = nn.Parameter(torch.ones(out_channels))
        self.theta = nn.Parameter(torch.zeros(out_channels))
        self.sigma = nn.Parameter(torch.ones(out_channels) * 1.0)
        self.gamma = nn.Parameter(torch.ones(out_channels) * 0.5)
        self.alpha = nn.Parameter(torch.ones(out_channels) * 0.5)

        self.conv_weights = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.conv_weights, mode="fan_out", nonlinearity="relu")

        self.bn = nn.BatchNorm2d(out_channels)

    def _build_gabor_kernel(self) -> torch.Tensor:
        sigma = torch.clamp(self.sigma, 0.1, 20.0)
        wavelength = torch.clamp(self.wavelength, 0.5, 50.0)
        gamma = torch.clamp(self.gamma, 0.1, 5.0)
        theta = self.theta

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        x_theta = self.grid_x.unsqueeze(0) * cos_theta.view(-1, 1, 1) + self.grid_y.unsqueeze(0) * sin_theta.view(-1, 1, 1)
        y_theta = -self.grid_x.unsqueeze(0) * sin_theta.view(-1, 1, 1) + self.grid_y.unsqueeze(0) * cos_theta.view(-1, 1, 1)

        gb = torch.exp(-(x_theta**2 + (gamma.view(-1, 1, 1) * y_theta) ** 2) / (2 * sigma.view(-1, 1, 1) ** 2))
        gr = torch.cos(2 * math.pi * x_theta / wavelength.view(-1, 1, 1))

        gabor = gb * gr

        alpha = torch.clamp(self.alpha, 0.1, 3.0)
        sign = torch.sign(gabor)
        gabor = sign * (torch.abs(gabor) + 1e-8).pow(alpha.view(-1, 1, 1))

        gabor = gabor / (gabor.view(self.out_channels, -1).norm(dim=1, keepdim=True).view(self.out_channels, 1, 1) + 1e-8)

        return gabor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gabor = self._build_gabor_kernel()
        combined = self.conv_weights * gabor.unsqueeze(1)
        out = F.conv2d(x, combined, padding=self.kernel_size // 2, groups=1)
        out = F.gelu(out)
        out = self.bn(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, H, W)
        out = self.proj(out)
        return out


class DepthwisePointwiseConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))
