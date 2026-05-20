"""Binary Panoptic Quality — matches LKCell's get_fast_pq exactly.

Uses pairwise IoU matrix + threshold 0.5 (no centroid distance filter).
Input: pred_masks (N, H, W) tensor of binary masks, gt_masks (M, H, W) tensor.
Internally converts to instance maps and calls get_fast_pq for LKCell-exact results.
"""
from __future__ import annotations

import torch
from torchmetrics import Metric

from wave_lk_cell.metrics.lkcell_metrics import get_fast_pq, remap_label


class BinaryPanopticQuality(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, match_iou: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.match_iou = match_iou
        self.add_state("pq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> None:
        """Update with binary instance masks.

        Args:
            pred_masks: (N, H, W) binary prediction masks.
            gt_masks: (M, H, W) binary ground-truth masks.
        """
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(0)

        for b in range(pred_masks.shape[0]):
            # Convert to (H, W) instance maps
            pred_map = self._masks_to_instance_map(pred_masks[b])
            gt_map = self._masks_to_instance_map(gt_masks[b])

            # Remap to contiguous labels (LKCell does this)
            pred_map = remap_label(pred_map)
            gt_map = remap_label(gt_map)

            [dq, sq, pq], _ = get_fast_pq(
                gt_map, pred_map, match_iou=self.match_iou,
            )

            self.pq_sum += pq
            self.dq_sum += dq
            self.sq_sum += sq
            self.count += 1

    @staticmethod
    def _masks_to_instance_map(masks: torch.Tensor) -> torch.Tensor:
        """Convert (N, H, W) binary masks → (H, W) instance map with labels 1..N."""
        N, H, W = masks.shape
        inst_map = torch.zeros(H, W, dtype=torch.int32)
        for i in range(N):
            inst_map[masks[i].bool()] = i + 1
        return inst_map

    def compute(self) -> dict[str, torch.Tensor]:
        count = self.count.item()
        if count == 0:
            return {"bPQ": torch.tensor(0.0), "bDQ": torch.tensor(0.0), "bSQ": torch.tensor(0.0)}
        return {
            "bPQ": self.pq_sum / count,
            "bDQ": self.dq_sum / count,
            "bSQ": self.sq_sum / count,
        }
