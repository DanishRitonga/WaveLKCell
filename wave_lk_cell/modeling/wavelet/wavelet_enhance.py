from __future__ import annotations

import torch
import torch.nn as nn

from wave_lk_cell.modeling.wavelet.dwt import DWT2, IDWT2
from wave_lk_cell.modeling.wavelet.processors import (
    AdaptivePowerGaborConv,
    ChannelAttention,
    DepthwisePointwiseConv,
    SelfAttention2d,
)


class MultiWaveletEnhance(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels

        self.dwt1 = DWT2(channels)
        self.dwt2 = DWT2(channels)

        self.hf_processor = AdaptivePowerGaborConv(channels, channels)
        self.mhf_processor = nn.Sequential(
            SelfAttention2d(channels, num_heads=num_heads),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.mid_processor = DepthwisePointwiseConv(channels)
        self.mlf_processor = nn.Sequential(
            ChannelAttention(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.lf_processor = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.idwt2 = IDWT2(channels)
        self.idwt1 = IDWT2(channels)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands1 = self.dwt1(x)
        bands2 = self.dwt2(bands1["LL"])

        hh1 = self.hf_processor(bands1["HH"])
        lh1 = self.mhf_processor(bands1["LH"])
        hl1 = self.mhf_processor(bands1["HL"])

        hh2 = self.mid_processor(bands2["HH"])
        lh2 = self.mlf_processor(bands2["LH"])
        hl2 = self.mlf_processor(bands2["HL"])
        ll2 = self.lf_processor(bands2["LL"])

        bands2_enhanced = {"LL": ll2, "LH": lh2, "HL": hl2, "HH": hh2}
        ll1_enhanced = self.idwt2(bands2_enhanced)

        bands1_enhanced = {"LL": ll1_enhanced, "LH": lh1, "HL": hl1, "HH": hh1}
        x_enhanced = self.idwt1(bands1_enhanced)

        return x + x_enhanced
