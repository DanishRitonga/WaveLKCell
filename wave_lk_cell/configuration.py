from __future__ import annotations

from transformers import PretrainedConfig


class WaveLKCellConfig(PretrainedConfig):
    model_type = "wave_lk_cell"

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
        decoder_channels: tuple[int, ...] = (1024, 512, 256, 128, 64),
        decoder_large_kernel_sizes: tuple[int, ...] = (13, 27, 29),
        decoder_small_kernel: int = 5,
        decoder_drop_path: float = 0.1,
        wavelet_num_heads: int = 4,
        pretrained_encoder_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_nuclei_classes = num_nuclei_classes
        self.num_tissue_classes = num_tissue_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.decoder_channels = decoder_channels
        self.decoder_large_kernel_sizes = decoder_large_kernel_sizes
        self.decoder_small_kernel = decoder_small_kernel
        self.decoder_drop_path = decoder_drop_path
        self.wavelet_num_heads = wavelet_num_heads
        self.pretrained_encoder_path = pretrained_encoder_path
