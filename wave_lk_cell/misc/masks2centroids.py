from __future__ import annotations

import torch


def masks2centroids(masks: torch.Tensor) -> torch.Tensor:
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    B, H, W = masks.shape
    centroids = torch.zeros(B, 2, device=masks.device, dtype=torch.float32)
    for i in range(B):
        nonzero = torch.nonzero(masks[i], as_tuple=False)
        if nonzero.numel() > 0:
            centroids[i, 0] = nonzero[:, 0].float().mean()
            centroids[i, 1] = nonzero[:, 1].float().mean()
        else:
            centroids[i, 0] = H / 2.0
            centroids[i, 1] = W / 2.0
    return centroids
