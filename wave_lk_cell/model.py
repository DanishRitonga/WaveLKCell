"""WaveLKCell model wrapper — matches LKCell's training pipeline.

Key changes to match LKCell:
- Loss uses XentropyLoss + DiceLoss for NP/Type, MSE + MSGE for HV, CE for tissue
- Targets are one-hot encoded for segmentation losses
- compute_loss takes device for MSGE
- Tissue classification loss added
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wave_lk_cell.losses import XentropyLoss, DiceLoss, MSELossMaps, MSGELossMaps
from wave_lk_cell.metrics import BinaryPanopticQuality, PanopticQuality
from wave_lk_cell.modeling import WaveLKCell
from wave_lk_cell.post_processing import post_process_batch


class NestedMetricCollection(nn.ModuleDict):
    def __init__(self, metric_cls: type, **kwargs) -> None:
        super().__init__()
        self._metric_cls = metric_cls
        self._kwargs = kwargs

    def __getitem__(self, key: str) -> Any:
        if key not in self:
            self[key] = self._metric_cls(**self._kwargs)
        return super().__getitem__(key)

    def update(self, key: str, *args, **kwargs) -> None:
        self[key].update(*args, **kwargs)

    def compute(self) -> dict[str, dict[str, torch.Tensor]]:
        return {k: v.compute() for k, v in self.items()}

    def reset(self) -> None:
        for v in self.values():
            v.reset()

    def __deepcopy__(self, memo):
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self._kwargs.items():
            setattr(new, f"_{k}", copy.deepcopy(v, memo))
        return new


class WaveLKCellModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        num_tissue_classes: int = 19,
        pretrained_encoder: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = WaveLKCell(
            num_nuclei_classes=num_classes,
            num_tissue_classes=num_tissue_classes,
            pretrained_encoder=pretrained_encoder,
        )
        self.encoder = self.model.encoder

        # Loss functions matching LKCell
        self.np_bce = XentropyLoss()
        self.np_dice = DiceLoss()
        self.hv_mse = MSELossMaps()
        self.hv_msge = MSGELossMaps()
        self.type_bce = XentropyLoss()
        self.type_dice = DiceLoss()
        self.tissue_ce = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss matching LKCell's 7-term loss."""
        device = outputs["nuclei_binary_map"].device
        w = weights or {}

        gt_binary = torch.stack([t["binary_map"] for t in targets]).to(device)
        gt_hv = torch.stack([t["hv_map"] for t in targets]).to(device)
        gt_type = torch.stack([t["type_map"] for t in targets]).to(device)

        # One-hot encode for XentropyLoss and DiceLoss
        gt_binary_onehot = F.one_hot(gt_binary.long(), num_classes=2).permute(0, 3, 1, 2).float()
        gt_type_onehot = F.one_hot(gt_type.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Softmax predictions for XentropyLoss and DiceLoss
        np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)
        type_pred = outputs["nuclei_type_map"].float().softmax(dim=1)

        # NP losses: BCE + Dice
        np_bce_loss = self.np_bce(np_pred, gt_binary_onehot)
        np_dice_loss = self.np_dice(np_pred, gt_binary_onehot)

        # HV losses: MSE + MSGE
        hv_pred = outputs["hv_map"].float()
        hv_mse_loss = self.hv_mse(hv_pred, gt_hv.float())
        hv_msge_loss = self.hv_msge(hv_pred, gt_hv.float(), gt_binary_onehot, str(device))

        # Type losses: BCE + Dice
        type_bce_loss = self.type_bce(type_pred, gt_type_onehot)
        type_dice_loss = self.type_dice(type_pred, gt_type_onehot)

        # Tissue loss (if tissue labels available)
        tissue_loss = torch.tensor(0.0, device=device)
        if "tissue_types" in outputs:
            tissue_labels = []
            for t in targets:
                tissue_idx = t.get("tissue_idx", 0)
                tissue_labels.append(tissue_idx)
            if tissue_labels:
                tissue_labels = torch.tensor(tissue_labels, dtype=torch.long, device=device)
                tissue_loss = self.tissue_ce(outputs["tissue_types"], tissue_labels)

        w_np = w.get("np_weight", 1.0)
        w_hv = w.get("hv_weight", 1.0)
        w_type = w.get("type_weight", 1.0)
        w_tissue = w.get("tissue_weight", 1.0)

        total = (
            w_np * (np_bce_loss + np_dice_loss)
            + w_hv * (hv_mse_loss + hv_msge_loss)
            + w_type * (type_bce_loss + type_dice_loss)
            + w_tissue * tissue_loss
        )

        return {
            "loss": total,
            "np_bce_loss": np_bce_loss,
            "np_dice_loss": np_dice_loss,
            "hv_mse_loss": hv_mse_loss,
            "hv_msge_loss": hv_msge_loss,
            "type_bce_loss": type_bce_loss,
            "type_dice_loss": type_dice_loss,
            "tissue_loss": tissue_loss,
        }

    def update_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        metrics: NestedMetricCollection,
    ) -> None:
        batch_size = outputs["nuclei_binary_map"].shape[0]
        np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)[:, 1]
        hv_pred = outputs["hv_map"].float()
        type_pred = outputs["nuclei_type_map"].float()

        for i in range(batch_size):
            tissue_key = str(targets[i]["tissue"])
            np_binary = (np_pred[i] > 0.5).cpu().numpy()
            hv_np = hv_pred[i].detach().cpu().numpy()
            gt_masks = targets[i]["masks"].cpu()
            gt_labels = targets[i]["labels"].cpu()

            pred_inst, pred_type = post_process_batch(
                np_binary[None], hv_np[None], type_pred[i:i+1].cpu().numpy(), self.num_classes,
            )[0]

            pred_mask_list = []
            pred_label_list = []
            for inst_id in np.unique(pred_inst):
                if inst_id == 0:
                    continue
                m = torch.from_numpy((pred_inst == inst_id).astype("float32"))
                pred_mask_list.append(m)
                if (pred_inst == inst_id).any():
                    pred_label_list.append(pred_type[pred_inst == inst_id][0])
                else:
                    pred_label_list.append(0)

            if pred_mask_list:
                pred_masks = torch.stack(pred_mask_list)
                pred_labels = torch.tensor(pred_label_list, dtype=torch.long)
            else:
                H, W = np_binary.shape
                pred_masks = torch.zeros(0, H, W)
                pred_labels = torch.zeros(0, dtype=torch.long)

            metrics.update(tissue_key, pred_masks, gt_masks, pred_labels, gt_labels)

    def update_binary_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        metrics: NestedMetricCollection,
    ) -> None:
        batch_size = outputs["nuclei_binary_map"].shape[0]
        np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)[:, 1]
        hv_pred = outputs["hv_map"].float()

        for i in range(batch_size):
            tissue_key = str(targets[i]["tissue"])
            np_binary = (np_pred[i] > 0.5).cpu().numpy()
            hv_np = hv_pred[i].detach().cpu().numpy()
            gt_masks = targets[i]["masks"].cpu()

            H, W = np_binary.shape
            pred_inst, _ = post_process_batch(
                np_binary[None], hv_np[None], np.zeros((1, self.num_classes, H, W)), self.num_classes,
            )[0]

            pred_mask_list = []
            for inst_id in np.unique(pred_inst):
                if inst_id == 0:
                    continue
                m = torch.from_numpy((pred_inst == inst_id).astype("float32"))
                pred_mask_list.append(m)

            if pred_mask_list:
                pred_masks = torch.stack(pred_mask_list)
            else:
                H, W = np_binary.shape
                pred_masks = torch.zeros(0, H, W)

            metrics.update(tissue_key, pred_masks, gt_masks)
