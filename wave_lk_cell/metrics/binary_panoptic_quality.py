from __future__ import annotations

import torch
import torchmetrics
from torchmetrics import Metric

from wave_lk_cell.misc.linear_assignment import linear_assignment_fn


class BinaryPanopticQuality(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("pq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("dq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sq_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> None:
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(0)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(0)

        pred_binary = pred_masks.sum(dim=1) > 0
        gt_binary = gt_masks.sum(dim=1) > 0

        intersection = (pred_binary & gt_binary).sum().float()
        union = (pred_binary | gt_binary).sum().float()
        dice = 2 * intersection / (union + 1e-8)

        pred_centroids = self._get_centroids(pred_masks)
        gt_centroids = self._get_centroids(gt_masks)

        tp, fp, fn = self._match_instances(pred_masks, gt_masks, pred_centroids, gt_centroids)

        dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-8)
        sq = self._compute_sq(pred_masks, gt_masks, tp)

        pq = dq * sq
        self.pq_sum += pq
        self.dq_sum += dq
        self.sq_sum += sq
        self.count += 1

    def _get_centroids(self, masks: torch.Tensor) -> torch.Tensor:
        B, N, H, W = masks.shape
        centroids = torch.zeros(B, N, 2, device=masks.device)
        for b in range(B):
            for n in range(N):
                nz = torch.nonzero(masks[b, n], as_tuple=False)
                if nz.numel() > 0:
                    centroids[b, n, 0] = nz[:, 0].float().mean()
                    centroids[b, n, 1] = nz[:, 1].float().mean()
                else:
                    centroids[b, n] = torch.tensor([H / 2.0, W / 2.0])
        return centroids

    def _match_instances(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_centroids: torch.Tensor,
        gt_centroids: torch.Tensor,
    ) -> tuple[float, float, float]:
        B, N_pred, H, W = pred_masks.shape
        N_gt = gt_masks.shape[1]
        tp = fp = fn = 0.0

        for b in range(B):
            if N_pred == 0 and N_gt == 0:
                continue
            elif N_pred == 0:
                fn += N_gt
                continue
            elif N_gt == 0:
                fp += N_pred
                continue

            dists = torch.cdist(pred_centroids[b], gt_centroids[b])
            matched = linear_assignment_fn(dists)

            matched_pred = set()
            matched_gt = set()
            for r, c in matched:
                if dists[r, c] < 20.0:
                    pred_pixels = pred_masks[b, r].bool()
                    gt_pixels = gt_masks[b, c].bool()
                    iou = (pred_pixels & gt_pixels).sum().float() / (pred_pixels | gt_pixels).sum().float() + 1e-8
                    if iou > 0.25:
                        tp += 1
                        matched_pred.add(r)
                        matched_gt.add(c)

            fp += N_pred - len(matched_pred)
            fn += N_gt - len(matched_gt)

        return tp, fp, fn

    def _compute_sq(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        tp: float,
    ) -> float:
        if tp < 1:
            return 0.0
        B, N_pred, H, W = pred_masks.shape
        N_gt = gt_masks.shape[1]
        iou_sum = 0.0
        matched_count = 0

        for b in range(B):
            pred_c = self._get_centroids(pred_masks[b:b+1])[0]
            gt_c = self._get_centroids(gt_masks[b:b+1])[0]
            dists = torch.cdist(pred_c, gt_c)
            matched = linear_assignment_fn(dists)
            for r, c in matched:
                if dists[r, c] < 20.0:
                    iou = (pred_masks[b, r] & gt_masks[b, c]).sum().float() / (pred_masks[b, r] | gt_masks[b, c]).sum().float() + 1e-8
                    if iou > 0.25:
                        iou_sum += iou.item()
                        matched_count += 1

        return iou_sum / (tp + 1e-8)

    def compute(self) -> dict[str, torch.Tensor]:
        count = self.count.item()
        if count == 0:
            return {"bPQ": torch.tensor(0.0), "bDQ": torch.tensor(0.0), "bSQ": torch.tensor(0.0)}
        return {
            "bPQ": self.pq_sum / count,
            "bDQ": self.dq_sum / count,
            "bSQ": self.sq_sum / count,
        }
