"""WaveLKCell — LKCell architecture with wavelet enhancement replacing stage 4.

Matches LKCell's CellViT(UniRepLKNet) architecture exactly except:
  - Encoder has 3 stages instead of 4 (depths (3,3,27) instead of (3,3,27,3))
  - Stage 4 is replaced by MultiWaveletEnhance + wavelet_downsample
  - Encoder uses independent input_conv/input_down_conv (not stem alias)

Everything else matches LKCell: decoder (UpCat/TwoConv), heads (BN→conv_bn→GELU→conv_bn),
tissue classification branch, output names.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from wave_lk_cell.modeling.lkcell_decoder import RepLKDecoder
from wave_lk_cell.modeling.lkcell_encoder import LKCellEncoder, LayerNormChannelsFirst
from wave_lk_cell.modeling.pretrained import load_unireplknet_s_encoder
from wave_lk_cell.modeling.wavelet import MultiWaveletEnhance

logger = logging.getLogger(__name__)


def _conv_bn(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Sequential:
    """Conv2d(bias=False) + BatchNorm2d — matches LKCell's conv_bn helper."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class SegmentationHead(nn.Sequential):
    """Exact match of LKCell's SegmentationHead.

    Structure: BN → Conv1x1+BN → GELU → Conv1x1+BN → Identity
    (no bias on convs, 1 GELU activation)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.BatchNorm2d(in_channels),
            _conv_bn(in_channels, in_channels, kernel_size=1),
            nn.GELU(),
            _conv_bn(in_channels, out_channels, kernel_size=1),
        )


class TissueClassifier(nn.Module):
    """Tissue type classification branch — GAP → Linear.

    Matches LKCell's tissue_types output (encoder-level global average pooling).
    """

    def __init__(self, in_channels: int, num_tissue_classes: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, num_tissue_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(x).flatten(1)
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

        # Wavelet enhancement replaces stage 4
        self.wavelet_enhance = MultiWaveletEnhance(dims[2], num_heads=4)
        self.wavelet_downsample = nn.Sequential(
            nn.Conv2d(dims[2], wavelet_out_channels, 3, stride=2, padding=1),
            LayerNormChannelsFirst(wavelet_out_channels),
        )

        # Encoder channels for decoder: (in_channels, *dims, wavelet_out_channels)
        encoder_channels = (in_channels,) + dims + (wavelet_out_channels,)

        # Decoder — matches LKCell's RepLKDeocder exactly
        self.decoder = RepLKDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

        # Segmentation heads — matching LKCell's SegmentationHead exactly
        self.np_head = SegmentationHead(decoder_channels[-1], 2)
        self.hv_head = SegmentationHead(decoder_channels[-1], 2)
        self.nt_head = SegmentationHead(decoder_channels[-1], num_nuclei_classes)

        # Tissue classification branch — matching LKCell
        self.tissue_classifier = TissueClassifier(wavelet_out_channels, num_tissue_classes)

        if pretrained_encoder:
            try:
                load_unireplknet_s_encoder(self.encoder, cache_dir=pretrained_encoder_path)
            except Exception:
                logger.warning("Could not load pretrained UniRepLKNet-S weights, training from scratch")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stage_features, input_features = self.encoder(x)

        # Wavelet bottleneck replaces stage 4
        enhanced = self.wavelet_enhance(stage_features[2])
        wavelet_feature = self.wavelet_downsample(enhanced)

        all_features = list(stage_features) + [wavelet_feature]
        decoder_out = self.decoder(all_features, input_features)

        # Tissue classification from wavelet bottleneck (analogous to LKCell's encoder stage 4 GAP)
        tissue_logits = self.tissue_classifier(wavelet_feature)

        return {
            "nuclei_binary_map": self.np_head(decoder_out),
            "hv_map": self.hv_head(decoder_out),
            "nuclei_type_map": self.nt_head(decoder_out),
            "tissue_types": tissue_logits,
        }
