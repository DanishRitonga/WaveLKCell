"""Multi-class Panoptic Quality — matches LKCell's per-class PQ exactly.

Uses get_fast_pq with match_iou=0.5 for each class independently,
then averages across classes — same as LKCell's calculate_step_metric_validation.
"""
from __future__ import annotations

import torch
from torchmetrics import Metric

from wave_lk_cell.metrics.lkcell_metrics import get_fast_pq, remap_label


class PanopticQuality(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, match_iou: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.match_iou = match_iou
        self.add_state("pq_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("dq_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("sq_sum", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_labels: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        """Update with instance masks + class labels.

        Args:
            pred_masks: (N, H, W) binary prediction instance masks.
            gt_masks: (M, H, W) binary ground-truth instance masks.
            pred_labels: (N,) class label per predicted instance.
            gt_labels: (M,) class label per ground-truth instance.
        """
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)
            pred_labels = pred_labels.unsqueeze(0)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0)

        B = pred_masks.shape[0]

        for b in range(B):
            for cls in range(self.num_classes):
                # Build per-class instance maps
                gt_cls_idx = (gt_labels[b] == cls).nonzero(as_tuple=True)[0]
                pred_cls_idx = (pred_labels[b] == cls).nonzero(as_tuple=True)[0]

                n_gt_cls = gt_cls_idx.numel()
                n_pred_cls = pred_cls_idx.numel()

                if n_gt_cls == 0 and n_pred_cls == 0:
                    continue  # skip — no instances of this class in either

                # Build instance maps for this class
                H, W = pred_masks.shape[-2], pred_masks.shape[-1]

                gt_cls_map = torch.zeros(H, W, dtype=torch.int32)
                for rank, idx in enumerate(gt_cls_idx):
                    gt_cls_map[gt_masks[b, idx].bool()] = rank + 1

                pred_cls_map = torch.zeros(H, W, dtype=torch.int32)
                for rank, idx in enumerate(pred_cls_idx):
                    pred_cls_map[pred_masks[b, idx].bool()] = rank + 1

                gt_cls_map = remap_label(gt_cls_map)
                pred_cls_map = remap_label(pred_cls_map)

                [dq, sq, pq], _ = get_fast_pq(
                    gt_cls_map, pred_cls_map, match_iou=self.match_iou,
                )

                self.pq_sum[cls] += pq
                self.dq_sum[cls] += dq
                self.sq_sum[cls] += sq
                self.count[cls] += 1

    def compute(self) -> dict[str, torch.Tensor]:
        count = self.count
        valid = count > 0
        pq = torch.where(valid, self.pq_sum / (count + 1e-8), torch.zeros_like(self.pq_sum))
        dq = torch.where(valid, self.dq_sum / (count + 1e-8), torch.zeros_like(self.dq_sum))
        sq = torch.where(valid, self.sq_sum / (count + 1e-8), torch.zeros_like(self.sq_sum))
        return {
            "mPQ": pq.mean(),
            "mDQ": dq.mean(),
            "mSQ": sq.mean(),
        }
