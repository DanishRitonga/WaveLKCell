from __future__ import annotations

import torch
from torchmetrics import Metric


class PanopticQuality(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
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
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)
            pred_labels = pred_labels.unsqueeze(0)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(0)
            gt_labels = gt_labels.unsqueeze(0)

        B, N_pred, H, W = pred_masks.shape
        N_gt = gt_masks.shape[1]

        for b in range(B):
            for cls in range(self.num_classes):
                gt_cls_mask = gt_labels[b] == cls
                pred_cls_mask = pred_labels[b] == cls

                n_gt_cls = gt_cls_mask.sum().item()
                n_pred_cls = pred_cls_mask.sum().item()

                if n_gt_cls == 0 and n_pred_cls == 0:
                    continue

                gt_cls_masks = gt_masks[b][gt_cls_mask]
                pred_cls_masks = pred_masks[b][pred_cls_mask]

                if gt_cls_masks.numel() == 0:
                    self.pq_sum[cls] += 0.0
                    self.count[cls] += 1
                    continue
                if pred_cls_masks.numel() == 0:
                    self.pq_sum[cls] += 0.0
                    self.count[cls] += 1
                    continue

                intersection = (pred_cls_masks.any(dim=0).bool() & gt_cls_masks.any(dim=0).bool()).sum().float()
                union = (pred_cls_masks.any(dim=0).bool() | gt_cls_masks.any(dim=0).bool()).sum().float()
                dice = 2 * intersection / (union + 1e-8)

                tp, fp, fn = self._match_by_class(pred_cls_masks, gt_cls_masks)
                dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
                sq = self._sq_by_class(pred_cls_masks, gt_cls_masks, tp)

                self.pq_sum[cls] += dq * sq
                self.dq_sum[cls] += dq
                self.sq_sum[cls] += sq
                self.count[cls] += 1

    def _match_by_class(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> tuple[float, float, float]:
        N_pred = pred_masks.shape[0]
        N_gt = gt_masks.shape[0]
        tp = fp = fn = 0.0

        for n in range(N_gt):
            best_iou = 0.0
            best_pred = -1
            for p in range(N_pred):
                inter = (pred_masks[p].bool() & gt_masks[n].bool()).sum().float()
                union = (pred_masks[p].bool() | gt_masks[n].bool()).sum().float()
                iou = inter / (union + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = p

            if best_iou > 0.5 and best_pred >= 0:
                tp += 1
            else:
                fn += 1

        fp = max(0, N_pred - int(tp))
        return tp, fp, fn

    def _sq_by_class(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, tp: float
    ) -> float:
        if tp < 1:
            return 0.0
        iou_sum = 0.0
        count = 0
        N_pred = pred_masks.shape[0]
        N_gt = gt_masks.shape[0]

        matched_pred = set()
        for n in range(N_gt):
            best_iou = 0.0
            best_pred = -1
            for p in range(N_pred):
                if p in matched_pred:
                    continue
                inter = (pred_masks[p].bool() & gt_masks[n].bool()).sum().float()
                union = (pred_masks[p].bool() | gt_masks[n].bool()).sum().float()
                iou = inter / (union + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = p
            if best_iou > 0.5 and best_pred >= 0:
                iou_sum += best_iou.item()
                count += 1
                matched_pred.add(best_pred)

        return iou_sum / (count + 1e-8)

    def compute(self) -> dict[str, torch.Tensor]:
        count = self.count
        valid = count > 0
        pq = torch.where(valid, self.pq_sum / (count + 1e-8), torch.zeros_like(self.pq_sum))
        return {
            "mPQ": pq.mean(),
            "mDQ": torch.where(valid, self.dq_sum / (count + 1e-8), torch.zeros_like(self.dq_sum)).mean(),
            "mSQ": torch.where(valid, self.sq_sum / (count + 1e-8), torch.zeros_like(self.sq_sum)).mean(),
        }
