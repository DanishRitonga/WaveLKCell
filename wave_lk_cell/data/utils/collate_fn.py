from __future__ import annotations

from typing import Any

import torch


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images = torch.stack([x[0] for x in batch])
    targets = [x[1] for x in batch]
    return images, targets
