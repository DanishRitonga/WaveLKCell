from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

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
        pretrained_encoder: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = WaveLKCell(
            num_nuclei_classes=num_classes,
            pretrained_encoder=pretrained_encoder,
        )
        self.encoder = self.model.encoder

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        w = weights or {}
        gt_binary = torch.stack([t["binary_map"] for t in targets]).to(outputs["nuclei_binary_map"].device)
        gt_hv = torch.stack([t["hv_map"] for t in targets]).to(outputs["hv_map"].device)
        gt_type = torch.stack([t["type_map"] for t in targets]).to(outputs["nuclei_type_map"].device)

        np_loss = F.cross_entropy(outputs["nuclei_binary_map"], gt_binary.long())
        hv_loss = F.mse_loss(outputs["hv_map"], gt_hv.float())
        type_loss = F.cross_entropy(outputs["nuclei_type_map"], gt_type)

        total = (
            w.get("np_weight", 1.0) * np_loss
            + w.get("hv_weight", 1.0) * hv_loss
            + w.get("type_weight", 1.0) * type_loss
        )
        return {"loss": total, "np_loss": np_loss, "hv_loss": hv_loss, "type_loss": type_loss}

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
            type_np = type_pred[i].cpu().numpy()
            gt_masks = targets[i]["masks"].cpu()
            gt_labels = targets[i]["labels"].cpu()

            pred_inst, pred_type = post_process_batch(
                np_binary[None], hv_np[None], type_pred[i:i+1].cpu().numpy(), self.num_classes,
            )[0]

            pred_mask_list = []
            pred_label_list = []
            for inst_id in __import__("numpy", fromlist=["unique"]).unique(pred_inst):
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
