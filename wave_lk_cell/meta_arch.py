from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning.pytorch as pl
from torchmetrics import Metric

from wave_lk_cell.configuration import WaveLKCellConfig
from wave_lk_cell.metrics import BinaryPanopticQuality, PanopticQuality
from wave_lk_cell.modeling import WaveLKCell
from wave_lk_cell.post_processing import post_process_batch


class NestedMetricCollection(nn.ModuleDict):
    def __init__(self, metric_cls: type[Metric], **kwargs) -> None:
        super().__init__()
        self._metric_cls = metric_cls
        self._kwargs = kwargs

    def __getitem__(self, key: str) -> Metric:
        if key not in self:
            self[key] = self._metric_cls(**self._kwargs)
        return super().__getitem__(key)

    def update(self, key: str, *args, **kwargs) -> None:
        metric = self[key]
        metric.update(*args, **kwargs)

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


class WaveLKCellMetaArch(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 5,
        warmup_epochs: int = 0,
        pretrained_encoder: bool = False,
        criterion: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        **config_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.criterion_config = criterion or {}
        self.optimizer_config = optimizer or {}
        self.scheduler_config = scheduler or {}

        self.config = WaveLKCellConfig(
            num_nuclei_classes=num_classes,
        )
        self.model = WaveLKCell(
            num_nuclei_classes=num_classes,
            pretrained_encoder=pretrained_encoder,
        )
        self.backbone = self.model.encoder

        self.val_metrics = NestedMetricCollection(BinaryPanopticQuality)
        self.test_binary_metrics = NestedMetricCollection(BinaryPanopticQuality)
        self.test_multiclass_metrics = NestedMetricCollection(
            PanopticQuality, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        gt_binary = torch.stack([t["binary_map"] for t in targets]).to(outputs["nuclei_binary_map"].device)
        gt_hv = torch.stack([t["hv_map"] for t in targets]).to(outputs["hv_map"].device)
        gt_type = torch.stack([t["type_map"] for t in targets]).to(outputs["nuclei_type_map"].device)

        np_loss = F.cross_entropy(outputs["nuclei_binary_map"], gt_binary.long())
        hv_loss = F.mse_loss(outputs["hv_map"], gt_hv.float())
        type_loss = F.cross_entropy(outputs["nuclei_type_map"], gt_type)

        np_weight = self.criterion_config.get("np_weight", 1.0)
        hv_weight = self.criterion_config.get("hv_weight", 1.0)
        type_weight = self.criterion_config.get("type_weight", 1.0)

        total_loss = (
            np_weight * np_loss
            + hv_weight * hv_loss
            + type_weight * type_loss
        )

        return {
            "loss": total_loss,
            "np_loss": np_loss,
            "hv_loss": hv_loss,
            "type_loss": type_loss,
        }

    def training_step(self, batch: tuple[torch.Tensor, list[dict]], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        outputs = self(images)
        losses = self._compute_loss(outputs, targets)

        for k, v in losses.items():
            self.log(f"train/{k}", v, prog_bar=(k == "loss"), batch_size=images.shape[0])

        return losses["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, list[dict]], batch_idx: int) -> None:
        images, targets = batch
        outputs = self(images)
        losses = self._compute_loss(outputs, targets)

        for k, v in losses.items():
            self.log(f"validation/{k}", v, batch_size=images.shape[0])

        self._update_instance_metrics(outputs, targets, "val")

    def test_step(self, batch: tuple[torch.Tensor, list[dict]], batch_idx: int) -> None:
        images, targets = batch
        outputs = self(images)
        self._update_instance_metrics(outputs, targets, "test")

    def _update_instance_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        stage: str,
    ) -> None:
        batch_size = outputs["nuclei_binary_map"].shape[0]

        np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)[:, 1]
        hv_pred = outputs["hv_map"].float()
        type_pred = outputs["nuclei_type_map"].float()

        for i in range(batch_size):
            tissue = targets[i]["tissue"]
            tissue_key = str(tissue)

            np_binary = (np_pred[i] > 0.5).cpu().numpy()
            hv_np = hv_pred[i].detach().cpu().numpy()
            type_np = type_pred[i].cpu().numpy()

            gt_masks = targets[i]["masks"].cpu()
            gt_labels = targets[i]["labels"].cpu()

            pred_inst, pred_type = post_process_batch(
                np_binary[None],
                hv_np[None],
                type_pred[i:i+1].cpu().numpy(),
                self.num_classes,
            )[0]

            pred_mask_list = []
            pred_label_list = []
            for inst_id in np.unique(pred_inst):
                if inst_id == 0:
                    continue
                m = torch.from_numpy((pred_inst == inst_id).astype(np.float32))
                pred_mask_list.append(m)
                pred_label_list.append(pred_type[pred_inst == inst_id][0] if (pred_inst == inst_id).any() else 0)

            if pred_mask_list:
                pred_masks = torch.stack(pred_mask_list)
                pred_labels = torch.tensor(pred_label_list, dtype=torch.long)
            else:
                H, W = np_binary.shape
                pred_masks = torch.zeros(0, H, W)
                pred_labels = torch.zeros(0, dtype=torch.long)

            self.val_metrics.update(tissue_key, pred_masks, gt_masks)
            self.test_binary_metrics.update(tissue_key, pred_masks, gt_masks)
            self.test_multiclass_metrics.update(tissue_key, pred_masks, gt_masks, pred_labels, gt_labels)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for tissue, vals in metrics.items():
            for k, v in vals.items():
                self.log(f"validation/{tissue}_{k}", v)
        avg_bpq = torch.stack([v["bPQ"] for v in metrics.values()]).mean()
        self.log("validation/bPQ", avg_bpq, prog_bar=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        binary_metrics = self.test_binary_metrics.compute()
        multiclass_metrics = self.test_multiclass_metrics.compute()

        for tissue, vals in binary_metrics.items():
            for k, v in vals.items():
                self.log(f"test/{tissue}_{k}", v)

        avg_bpq = torch.stack([v["bPQ"] for v in binary_metrics.values()]).mean()
        self.log("test/bPQ", avg_bpq)

        for tissue, vals in multiclass_metrics.items():
            for k, v in vals.items():
                self.log(f"test/{tissue}_{k}", v)

        self.test_binary_metrics.reset()
        self.test_multiclass_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        lr = self.optimizer_config.get("lr", 8e-4)
        betas = self.optimizer_config.get("betas", [0.85, 0.95])
        weight_decay = self.optimizer_config.get("weight_decay", 0.05)

        backbone_params = list(self.backbone.parameters())
        other_params = [p for p in self.parameters() if not any(p is bp for bp in backbone_params)]
        param_groups = [
            {"params": other_params},
            {"params": backbone_params, "lr": lr * 0.1},
        ]
        optimizer = AdamW(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)

        eta_min = self.scheduler_config.get("eta_min", 1e-5)
        t_max = self.scheduler_config.get("T_max", None)
        max_epochs = t_max if t_max else 130
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
