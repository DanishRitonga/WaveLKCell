from __future__ import annotations

import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

UNIREPLKNET_S_IN22K_FILE = "unireplknet_s_in22k_pretrain.pth"
REPO_ID = "DingXiaoH/UniRepLKNet"


def load_unireplknet_s_encoder(
    encoder: torch.nn.Module,
    cache_dir: str | Path | None = None,
) -> None:
    cache_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=UNIREPLKNET_S_IN22K_FILE,
        cache_dir=cache_dir,
    )
    state_dict = torch.load(cache_path, map_location="cpu")

    encoder_state = {}
    prefix_map = {
        "downsample_layers": "downsample_layers",
        "stages": "stages",
        "norms": "norms",
    }
    for key, value in state_dict.items():
        if key.startswith("head."):
            continue
        for prefix in prefix_map:
            if key.startswith(prefix):
                stage_idx = _extract_stage_index(key, prefix)
                if stage_idx is not None and stage_idx < 3:
                    encoder_state[key] = value
                elif stage_idx is None:
                    if "input_conv" not in key and "input_down_conv" not in key:
                        if prefix == "downsample_layers":
                            ds_idx = _extract_downsample_index(key)
                            if ds_idx is not None and ds_idx < 3:
                                encoder_state[key] = value
                break

    msg = encoder.load_state_dict(encoder_state, strict=False)
    logger.info(f"Loaded UniRepLKNet-S pretrained encoder: {msg}")


def _extract_stage_index(key: str, prefix: str) -> int | None:
    remainder = key[len(prefix) + 1:]
    for i in range(10):
        if remainder.startswith(f"{i}."):
            return i
    return None


def _extract_downsample_index(key: str) -> int | None:
    remainder = key[len("downsample_layers."):]
    for i in range(10):
        if remainder.startswith(f"{i}."):
            return i
    return None
