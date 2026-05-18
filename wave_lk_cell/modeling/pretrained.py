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
    for key, value in state_dict.items():
        if key.startswith("head."):
            continue
        new_key = _remap_key(key)
        if new_key is not None:
            encoder_state[new_key] = value

    msg = encoder.load_state_dict(encoder_state, strict=False)
    logger.info(f"Loaded UniRepLKNet-S pretrained encoder: {msg}")


def _remap_key(key: str) -> str | None:
    if key.startswith("stages."):
        idx = int(key.split(".")[1])
        if idx < 3:
            return key
        return None

    if key.startswith("downsample_layers."):
        parts = key.split(".")
        idx = int(parts[1])
        if idx == 0:
            return key
        if idx < 3:
            rest = ".".join(parts[2:])
            if rest.startswith("0."):
                return f"downsample_layers.{idx}.conv.{rest[2:]}"
            if rest.startswith("1."):
                return f"downsample_layers.{idx}.norm.{rest[2:]}"
            return f"downsample_layers.{idx}.{rest}"
        return None

    if key.startswith("norms."):
        idx = int(key.split(".")[1])
        if idx < 3:
            return key
        return None

    return None
