from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_lk_cell.modeling.encoder import UniRepLKNet
from wave_lk_cell.modeling.decoder import RepLKDecoder
from wave_lk_cell.modeling.wavelet.wavelet_enhance import MultiWaveletEnhance
from wave_lk_cell.post_processing import DetectionCellPostProcessor


def _conv_bn(
    in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        bn_layer = nn.BatchNorm2d(in_channels)
        conv_layer1 = _conv_bn(in_channels, in_channels, kernel_size=1)
        nonlinear_layer = nn.GELU()
        conv_layer2 = _conv_bn(in_channels, out_channels, kernel_size=1)
        super().__init__(bn_layer, conv_layer1, nonlinear_layer, conv_layer2)


@dataclass
class DataclassHVStorage:
    nuclei_binary_map: torch.Tensor
    hv_map: torch.Tensor
    nuclei_type_map: torch.Tensor
    tissue_types: torch.Tensor
    instance_map: torch.Tensor | None = None
    instance_types: list | None = None
    instance_types_nuclei: torch.Tensor | None = None
    batch_size: int = 0
    regression_map: torch.Tensor | None = None
    num_nuclei_classes: int = 6

    def get_dict(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and v is not None
        }


class WaveLKCell(nn.Module):
    def __init__(
        self,
        num_nuclei_classes: int = 6,
        num_tissue_classes: int = 19,
        pretrained_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_nuclei_classes = num_nuclei_classes
        self.num_tissue_classes = num_tissue_classes

        dims = (96, 192, 384, 768)
        self.encoder = UniRepLKNet(
            in_chans=3,
            depths=(3, 3, 27, 3),
            dims=dims,
            drop_path_rate=0.3,
            layer_scale_init_value=1e-6,
        )

        self.wavelet_enhance = MultiWaveletEnhance(dims[2])
        self.wavelet_downsample = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[3]),
        )

        encoder_channels = (3, dims[0], dims[1], dims[2], dims[3])
        decoder_channels = (1024, 512, 256, 128, 64)
        self.decoder = RepLKDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

        self.nuclei_binary_segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=2,
        )
        self.hv_map_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=2,
        )
        self.nuclei_type_maps_head = SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=self.num_nuclei_classes,
        )

        self.tissue_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[3]),
            nn.Linear(dims[3], num_tissue_classes),
        )

        if pretrained_encoder:
            self._load_pretrained_encoder()

    def _load_pretrained_encoder(self) -> None:
        from wave_lk_cell.modeling.pretrained import load_unireplknet_s_encoder
        load_unireplknet_s_encoder(self.encoder)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features, input_features, raw_features = self.encoder(x)

        wavelet_out = self.wavelet_enhance(raw_features[2])
        wavelet_feat = self.wavelet_downsample(wavelet_out)
        features[3] = self.encoder.norm3(wavelet_feat)

        decoder_out = self.decoder(features, input_features, skip_connect=3)

        nuclei_binary_map = self.nuclei_binary_segmentation_head(decoder_out)
        hv_map = self.hv_map_head(decoder_out)
        nuclei_type_map = self.nuclei_type_maps_head(decoder_out)
        tissue_types = self.tissue_classifier(features[3])

        return {
            "nuclei_binary_map": nuclei_binary_map,
            "hv_map": hv_map,
            "nuclei_type_map": nuclei_type_map,
            "tissue_types": tissue_types,
        }

    def calculate_instance_map(
        self,
        predictions: dict[str, torch.Tensor],
        magnification: int = 40,
    ) -> Tuple[torch.Tensor, list[dict]]:
        np_map = predictions["nuclei_binary_map"].permute(0, 2, 3, 1)
        tp_map = predictions["nuclei_type_map"].permute(0, 2, 3, 1)
        hv_map = predictions["hv_map"].permute(0, 2, 3, 1)

        batch_size = predictions["nuclei_binary_map"].shape[0]
        post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes,
            magnification=magnification,
            gt=False,
        )

        instance_maps = []
        instance_types = []
        for i in range(batch_size):
            pred_type = torch.argmax(tp_map[i], dim=-1).unsqueeze(-1)
            pred_binary = torch.argmax(np_map[i], dim=-1).unsqueeze(-1)
            pred_hv = hv_map[i]

            pred_map = torch.cat([pred_type, pred_binary, pred_hv], dim=-1)
            pred_map_np = pred_map.detach().cpu().numpy().astype(np.float32)

            inst_map, inst_dict = post_processor.post_process_cell_segmentation(pred_map_np)
            instance_maps.append(torch.from_numpy(inst_map))
            instance_types.append(inst_dict)

        instance_maps = torch.stack(instance_maps).to(np_map.device)
        return instance_maps, instance_types

    def generate_instance_nuclei_map(
        self,
        instance_map: torch.Tensor,
        instance_types: list[dict],
    ) -> torch.Tensor:
        batch_size = instance_map.shape[0]
        h, w = instance_map.shape[1], instance_map.shape[2]
        nuclei_instance_map = torch.zeros(
            (batch_size, self.num_nuclei_classes, h, w),
            dtype=torch.int32, device=instance_map.device,
        )

        for i in range(batch_size):
            for inst_id, inst_info in instance_types[i].items():
                inst_type = inst_info["type"]
                if inst_type is not None and inst_type > 0:
                    mask = instance_map[i] == inst_id
                    nuclei_instance_map[i, inst_type][mask] = inst_id

        return nuclei_instance_map

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
