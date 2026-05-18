from __future__ import annotations

import logging

import torch
import torch.nn as nn

from wave_lk_cell.modeling.lkcell_decoder import LKCellDecoder
from wave_lk_cell.modeling.lkcell_encoder import LKCellEncoder, LayerNormChannelsFirst
from wave_lk_cell.modeling.pretrained import load_unireplknet_s_encoder
from wave_lk_cell.modeling.wavelet import MultiWaveletEnhance

logger = logging.getLogger(__name__)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class WaveLKCell(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_nuclei_classes: int = 5,
        num_tissue_classes: int = 19,
        depths: tuple[int, ...] = (3, 3, 27),
        dims: tuple[int, ...] = (96, 192, 384),
        wavelet_out_channels: int = 768,
        drop_path_rate: float = 0.3,
        layer_scale_init_value: float = 1e-6,
        encoder_kernel_sizes: tuple[tuple[int, ...], ...] | None = None,
        decoder_channels: tuple[int, ...] = (1024, 512, 256, 128, 64),
        decoder_large_kernel_sizes: tuple[int, ...] = (13, 27, 29),
        decoder_small_kernel: int = 5,
        decoder_drop_path: float = 0.1,
        wavelet_num_heads: int = 4,
        pretrained_encoder: bool = False,
        pretrained_encoder_path: str | None = None,
    ) -> None:
        super().__init__()
        self.num_nuclei_classes = num_nuclei_classes
        self.num_tissue_classes = num_tissue_classes

        self.encoder = LKCellEncoder(
            in_channels=in_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            kernel_sizes=encoder_kernel_sizes,
        )

        self.wavelet_enhance = MultiWaveletEnhance(dims[2], num_heads=wavelet_num_heads)
        self.wavelet_downsample = nn.Sequential(
            nn.Conv2d(dims[2], wavelet_out_channels, 3, stride=2, padding=1),
            LayerNormChannelsFirst(wavelet_out_channels),
        )

        full_dims = dims + (wavelet_out_channels,)
        self.decoder = LKCellDecoder(
            encoder_dims=full_dims,
            decoder_channels=decoder_channels,
            large_kernel_sizes=decoder_large_kernel_sizes,
            small_kernel=decoder_small_kernel,
            drop_path=decoder_drop_path,
        )

        self.np_head = SegmentationHead(decoder_channels[-1], 2)
        self.hv_head = SegmentationHead(decoder_channels[-1], 2)
        self.nt_head = SegmentationHead(decoder_channels[-1], num_nuclei_classes)

        self.tissue_head = nn.Linear(wavelet_out_channels, num_tissue_classes)

        if pretrained_encoder:
            try:
                load_unireplknet_s_encoder(self.encoder, cache_dir=pretrained_encoder_path)
            except Exception:
                logger.warning("Could not load pretrained UniRepLKNet-S weights, training from scratch")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage_features, input_features = self.encoder(x)

        enhanced = self.wavelet_enhance(stage_features[2])
        wavelet_feature = self.wavelet_downsample(enhanced)

        tissue_logits = self.tissue_head(wavelet_feature.mean(dim=[-2, -1]))

        all_features = list(stage_features) + [wavelet_feature]
        decoder_out = self.decoder(all_features, input_features)

        return {
            "tissue_types": tissue_logits,
            "nuclei_binary_map": self.np_head(decoder_out),
            "hv_map": self.hv_head(decoder_out),
            "nuclei_type_map": self.nt_head(decoder_out),
        }
