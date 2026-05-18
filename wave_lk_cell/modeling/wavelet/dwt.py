from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        kernels = torch.stack([ll, lh, hl, hh])
        kernels = kernels.unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer("weight", kernels)
        self.groups = in_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Input spatial dims must be even, got {H}x{W}"
        out = F.conv2d(x, self.weight, stride=2, groups=C, padding=0)
        ll, lh, hl, hh = out[:, 0::4, :, :], out[:, 1::4, :, :], out[:, 2::4, :, :], out[:, 3::4, :, :]
        return ll, lh, hl, hh


class HaarIDWT2d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)
        kernel = torch.stack([ll, lh, hl, hh])
        kernel = kernel.unsqueeze(1)
        self.register_buffer("weight", kernel)

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = ll.shape
        x = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, H, W)
        out = F.conv_transpose2d(x, self.weight, stride=2, groups=1, padding=0)
        return out.reshape(B, C, H * 2, W * 2)


class DWT2(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.dwt = HaarDWT2d(in_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ll, lh, hl, hh = self.dwt(x)
        return {"LL": ll, "LH": lh, "HL": hl, "HH": hh}


class IDWT2(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.idwt = HaarIDWT2d(in_channels)

    def forward(self, bands: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.idwt(bands["LL"], bands["LH"], bands["HL"], bands["HH"])
