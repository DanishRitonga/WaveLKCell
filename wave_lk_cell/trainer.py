from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from wave_lk_cell.model import WaveLKCell, DataclassHVStorage
from wave_lk_cell.losses import (
    XentropyLoss,
    DiceLoss,
    MSELossMaps,
    MSGELossMaps,
    FocalTverskyLoss,
    MCFocalTverskyLoss,
)
from wave_lk_cell.metrics import get_fast_pq, remap_label

logger = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if np.isnan(val) or np.isinf(val):
            return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience: int = 130, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class WaveLKCellTrainer:
    def __init__(
        self,
        model: WaveLKCell,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        loss_fn_dict: dict | None = None,
        lr: float = 8e-4,
        backbone_lr_ratio: float = 0.1,
        betas: tuple[float, float] = (0.85, 0.95),
        weight_decay: float = 0.05,
        epochs: int = 130,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 1.0,
        eta_min: float = 1e-5,
        unfreeze_epoch: int = 25,
        patience: int = 130,
        magnification: int = 40,
        amp: bool = True,
        num_classes: int = 6,
        save_dir: str = "runs/wavellkcell",
        experiment_name: str = "wavellkcell",
        dataset_config: dict | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.accumulate = accumulate_grad_batches
        self.clip_val = gradient_clip_val
        self.unfreeze_epoch = unfreeze_epoch
        self.magnification = magnification
        self.mixed_precision = amp and self.device.type == "cuda"
        self.num_nuclei_classes = num_classes
        self.loss_fn_dict = loss_fn_dict or self._default_loss_dict()
        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        self.early_stopping = EarlyStopping(patience=patience, mode="max")
        self.dataset_config = dataset_config or {}

        self.model.freeze_encoder()
        backbone_params = list(self.model.encoder.parameters())
        other_params = [p for p in self.model.parameters() if not any(p is bp for bp in backbone_params)]
        self.optimizer = AdamW(
            [
                {"params": other_params},
                {"params": backbone_params, "lr": lr * backbone_lr_ratio},
            ],
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=eta_min)

        self.loss_avg_tracker: dict[str, AverageMeter] = {
            "Total_Loss": AverageMeter("Total_Loss"),
        }
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"] = AverageMeter(f"{branch}_{loss_name}")

        self.batch_avg_tissue_acc = AverageMeter("Tissue_ACC")
        self.best_fitness = 0.0
        self.start_epoch = 0

    @staticmethod
    def _default_loss_dict() -> dict:
        return {
            "nuclei_binary_map": {
                "focaltverskyloss": {"loss_fn": FocalTverskyLoss(), "weight": 1.0},
                "dice": {"loss_fn": DiceLoss(), "weight": 1.0},
            },
            "hv_map": {
                "mse": {"loss_fn": MSELossMaps(), "weight": 2.5},
                "msge": {"loss_fn": MSGELossMaps(), "weight": 8.0},
            },
            "nuclei_type_map": {
                "bce": {"loss_fn": XentropyLoss(), "weight": 0.5},
                "dice": {"loss_fn": DiceLoss(), "weight": 0.2},
                "mcfocaltverskyloss": {
                    "loss_fn": MCFocalTverskyLoss(num_classes=6),
                    "weight": 0.5,
                },
            },
            "tissue_types": {
                "ce": {"loss_fn": nn.CrossEntropyLoss(), "weight": 0.1},
            },
        }

    @staticmethod
    def _to_float32(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                out[k] = v.float()
            else:
                out[k] = v
        return out

    def calculate_loss(self, predictions: dict, gt: dict) -> torch.Tensor:
        predictions = self._to_float32(predictions)
        gt = self._to_float32(gt)
        total_loss = 0
        for branch, pred in predictions.items():
            if branch in ["instance_map", "instance_types", "instance_types_nuclei"]:
                continue
            if branch not in self.loss_fn_dict:
                continue
            for loss_name, loss_setting in self.loss_fn_dict[branch].items():
                loss_fn = loss_setting["loss_fn"]
                weight = loss_setting["weight"]
                if loss_name == "msge":
                    loss_value = loss_fn(
                        input=pred, target=gt[branch],
                        focus=gt["nuclei_binary_map"], device=self.device,
                    )
                else:
                    loss_value = loss_fn(input=pred, target=gt[branch])
                total_loss = total_loss + weight * loss_value
                self.loss_avg_tracker[f"{branch}_{loss_name}"].update(
                    loss_value.detach().cpu().numpy()
                )
        self.loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())
        return total_loss

    def unpack_predictions(self, predictions: dict) -> DataclassHVStorage:
        predictions["tissue_types"] = predictions["tissue_types"].to(self.device)
        predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
        predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
        (
            predictions["instance_map"],
            predictions["instance_types"],
        ) = self.model.calculate_instance_map(predictions, self.magnification)
        predictions["instance_types_nuclei"] = self.model.generate_instance_nuclei_map(
            predictions["instance_map"], predictions["instance_types"],
        ).to(self.device)
        if "regression_map" not in predictions:
            predictions["regression_map"] = None
        return DataclassHVStorage(
            nuclei_binary_map=predictions["nuclei_binary_map"],
            hv_map=predictions["hv_map"],
            nuclei_type_map=predictions["nuclei_type_map"],
            tissue_types=predictions["tissue_types"],
            instance_map=predictions["instance_map"],
            instance_types=predictions["instance_types"],
            instance_types_nuclei=predictions["instance_types_nuclei"],
            batch_size=predictions["tissue_types"].shape[0],
            regression_map=predictions["regression_map"],
            num_nuclei_classes=self.num_nuclei_classes,
        )

    def unpack_masks(self, masks: dict, tissue_types: list) -> DataclassHVStorage:
        gt_nuclei_binary_map_onehot = F.one_hot(
            masks["nuclei_binary_map"].long(), num_classes=2
        ).float()
        nuclei_type_maps = masks["nuclei_type_map"].long()
        gt_nuclei_type_maps_onehot = F.one_hot(
            nuclei_type_maps, num_classes=self.num_nuclei_classes
        ).float()

        gt = {
            "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(self.device),
            "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(self.device),
            "hv_map": masks["hv_map"].to(self.device),
            "instance_map": masks["instance_map"].to(self.device),
            "instance_types_nuclei": (
                gt_nuclei_type_maps_onehot * masks["instance_map"][..., None]
            ).permute(0, 3, 1, 2).to(self.device),
            "tissue_types": torch.tensor(
                [self._tissue_to_idx(t) for t in tissue_types],
                dtype=torch.long, device=self.device,
            ),
        }
        return DataclassHVStorage(
            **gt,
            batch_size=gt["tissue_types"].shape[0],
            num_nuclei_classes=self.num_nuclei_classes,
        )

    def _tissue_to_idx(self, tissue: str | int) -> int:
        if isinstance(tissue, int):
            return tissue
        tissue_map = self.dataset_config.get("tissue_types", {})
        if tissue in tissue_map:
            return tissue_map[tissue]
        return 0

    def _unpack_batch(self, batch):
        imgs = batch[0].to(self.device)
        targets = batch[1]

        masks_dict = {}
        masks_dict["nuclei_binary_map"] = torch.stack([t["binary_map"] for t in targets]).long()
        masks_dict["hv_map"] = torch.stack([t["hv_map"] for t in targets]).float()
        masks_dict["nuclei_type_map"] = torch.stack([t["type_map"] for t in targets]).long()

        instance_maps = []
        for t in targets:
            inst_map = torch.zeros_like(t["binary_map"], dtype=torch.int32)
            m = t["masks"]
            for j in range(m.shape[0]):
                inst_map[m[j] > 0] = j + 1
            instance_maps.append(inst_map)
        masks_dict["instance_map"] = torch.stack(instance_maps)

        tissue_types = [t.get("tissue", "unknown") for t in targets]

        return imgs, masks_dict, tissue_types

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        if epoch >= self.unfreeze_epoch:
            self.model.unfreeze_encoder()

        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        self.batch_avg_tissue_acc.reset()

        binary_dice_scores = []
        tissue_pred = []
        tissue_gt = []

        train_loop = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader))

        last_opt_step = -1
        for batch_idx, batch in train_loop:
            imgs, masks_dict, tissue_types = self._unpack_batch(batch)

            try:
                if self.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions_ = self.model(imgs)
                    predictions = self.unpack_predictions(predictions_)
                    gt = self.unpack_masks(masks_dict, tissue_types)
                    total_loss = self.calculate_loss(predictions.get_dict(), gt.get_dict())
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    self.scaler.scale(total_loss / self.accumulate).backward()
                else:
                    predictions_ = self.model(imgs)
                    predictions = self.unpack_predictions(predictions_)
                    gt = self.unpack_masks(masks_dict, tissue_types)
                    total_loss = self.calculate_loss(predictions.get_dict(), gt.get_dict())
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                    (total_loss / self.accumulate).backward()

                if batch_idx == 0 and epoch < 3:
                    np_raw = predictions_["nuclei_binary_map"].detach()
                    np_soft = F.softmax(np_raw.float(), dim=1)
                    fg_prob = np_soft[:, 1].mean().item()
                    bg_prob = np_soft[:, 0].mean().item()
                    fg_argmax = (torch.argmax(np_raw, dim=1) == 1).float().mean().item()
                    print(f"  [train ep{epoch}] NP logits: fg_prob={fg_prob:.4f} bg_prob={bg_prob:.4f} argmax_fg={fg_argmax:.4f}")

                if (batch_idx - last_opt_step) >= self.accumulate:
                    last_opt_step = batch_idx
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            pred_dict = predictions.get_dict()
            gt_dict = gt.get_dict()
            pred_tissue = torch.argmax(F.softmax(pred_dict["tissue_types"], dim=-1), dim=-1).cpu().numpy().astype(np.uint8)
            gt_tissue = gt_dict["tissue_types"].cpu().numpy().astype(np.uint8)
            tissue_acc = accuracy_score(y_true=gt_tissue, y_pred=pred_tissue)
            self.batch_avg_tissue_acc.update(tissue_acc)
            tissue_pred.append(pred_tissue)
            tissue_gt.append(gt_tissue)

            for i in range(len(pred_tissue)):
                pred_binary_map = torch.argmax(pred_dict["nuclei_binary_map"][i], dim=0)
                target_binary_map = torch.argmax(gt_dict["nuclei_binary_map"][i], dim=0).type(torch.uint8)
                intersection = (pred_binary_map * target_binary_map).sum().float()
                union = pred_binary_map.sum().float() + target_binary_map.sum().float()
                cell_dice = (2 * intersection + 1e-8) / (union + 1e-8)
                binary_dice_scores.append(float(cell_dice.detach().cpu()))

            train_loop.set_postfix({
                "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                "Dice": np.round(np.nanmean(binary_dice_scores), 3),
            })

        return {
            "Loss/Train": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Train": np.nanmean(binary_dice_scores),
            "Tissue-Multiclass-Accuracy/Train": accuracy_score(
                y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
            ),
        }

    @torch.no_grad()
    def validation_epoch(self, epoch: int) -> tuple[dict, float]:
        self.model.eval()

        self.loss_avg_tracker["Total_Loss"].reset()
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                self.loss_avg_tracker[f"{branch}_{loss_name}"].reset()
        self.batch_avg_tissue_acc.reset()

        binary_dice_scores = []
        pq_scores = []
        cell_type_pq_scores = []
        tissue_pred = []
        tissue_gt = []

        val_loop = tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader))

        for batch_idx, batch in val_loop:
            imgs, masks_dict, tissue_types = self._unpack_batch(batch)

            if self.mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions_ = self.model(imgs)
                predictions = self.unpack_predictions(predictions_)
                gt = self.unpack_masks(masks_dict, tissue_types)
                _ = self.calculate_loss(predictions.get_dict(), gt.get_dict())
            else:
                predictions_ = self.model(imgs)
                predictions = self.unpack_predictions(predictions_)
                gt = self.unpack_masks(masks_dict, tissue_types)
                _ = self.calculate_loss(predictions.get_dict(), gt.get_dict())

            if batch_idx == 0 and epoch < 3:
                np_raw = predictions_["nuclei_binary_map"].detach()
                np_soft = F.softmax(np_raw.float(), dim=1)
                fg_prob = np_soft[:, 1].mean().item()
                bg_prob = np_soft[:, 0].mean().item()
                fg_argmax = (torch.argmax(np_raw, dim=1) == 1).float().mean().item()
                print(f"  [val ep{epoch}] NP logits: fg_prob={fg_prob:.4f} bg_prob={bg_prob:.4f} argmax_fg={fg_argmax:.4f}")

            pred_dict = predictions.get_dict()
            gt_dict = gt.get_dict()
            pred_tissue = torch.argmax(F.softmax(pred_dict["tissue_types"], dim=-1), dim=-1).cpu().numpy().astype(np.uint8)
            gt_tissue = gt_dict["tissue_types"].cpu().numpy().astype(np.uint8)
            self.batch_avg_tissue_acc.update(accuracy_score(y_true=gt_tissue, y_pred=pred_tissue))
            tissue_pred.append(pred_tissue)
            tissue_gt.append(gt_tissue)

            pred_instance_types_nuclei = pred_dict["instance_types_nuclei"].cpu().numpy().astype(np.int32)
            gt_instance_maps = gt_dict["instance_map"].cpu()
            gt_instance_types_nuclei = gt_dict["instance_types_nuclei"].cpu().numpy().astype(np.int32)

            for i in range(len(pred_tissue)):
                pred_binary_map = torch.argmax(pred_dict["nuclei_binary_map"][i], dim=0)
                target_binary_map = torch.argmax(gt_dict["nuclei_binary_map"][i], dim=0).type(torch.uint8)
                intersection = (pred_binary_map * target_binary_map).sum().float()
                union = pred_binary_map.sum().float() + target_binary_map.sum().float()
                cell_dice = (2 * intersection + 1e-8) / (union + 1e-8)
                binary_dice_scores.append(float(cell_dice.detach().cpu()))

                remapped_instance_pred = remap_label(pred_dict["instance_map"][i].cpu())
                remapped_gt = remap_label(gt_instance_maps[i])
                [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_instance_pred)
                pq_scores.append(pq)

                nuclei_type_pq = []
                for j in range(self.num_nuclei_classes):
                    pred_nuclei_instance_class = remap_label(pred_instance_types_nuclei[i][j, ...])
                    target_nuclei_instance_class = remap_label(gt_instance_types_nuclei[i][j, ...])
                    if len(np.unique(target_nuclei_instance_class)) == 1:
                        pq_tmp = np.nan
                    else:
                        [_, _, pq_tmp], _ = get_fast_pq(pred_nuclei_instance_class, target_nuclei_instance_class, match_iou=0.5)
                    nuclei_type_pq.append(pq_tmp)
                cell_type_pq_scores.append(nuclei_type_pq)

            val_loop.set_postfix({
                "Loss": np.round(self.loss_avg_tracker["Total_Loss"].avg, 3),
                "Dice": np.round(np.nanmean(binary_dice_scores), 3),
                "bPQ": np.round(np.nanmean(pq_scores), 3),
            })

        scalar_metrics = {
            "Loss/Validation": self.loss_avg_tracker["Total_Loss"].avg,
            "Binary-Cell-Dice-Mean/Validation": np.nanmean(binary_dice_scores),
            "Tissue-Multiclass-Accuracy/Validation": accuracy_score(
                y_true=np.concatenate(tissue_gt), y_pred=np.concatenate(tissue_pred)
            ),
            "bPQ/Validation": np.nanmean(pq_scores),
            "mPQ/Validation": np.nanmean([np.nanmean(pq) for pq in cell_type_pq_scores]),
        }
        for branch, loss_fns in self.loss_fn_dict.items():
            for loss_name in loss_fns:
                scalar_metrics[f"{branch}_{loss_name}/Validation"] = self.loss_avg_tracker[f"{branch}_{loss_name}"].avg

        return scalar_metrics, np.nanmean(pq_scores)

    def fit(self) -> dict:
        print(f"{'Epoch':>5} {'loss':>10} {'Dice':>10} {'bPQ':>10} {'mPQ':>10} {'lr':>12}")
        print("-" * 70)

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)

            val_metrics = {}
            fitness = 0.0
            if self.val_loader is not None:
                torch.cuda.empty_cache()
                val_metrics, fitness = self.validation_epoch(epoch)

            is_best = fitness > self.best_fitness
            if is_best:
                self.best_fitness = fitness

            self._save_checkpoint(epoch, is_best)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"{epoch+1:>5} "
                f"{train_metrics.get('Loss/Train', 0):>10.4f} "
                f"{val_metrics.get('Binary-Cell-Dice-Mean/Validation', train_metrics.get('Binary-Cell-Dice-Mean/Train', 0)):>10.4f} "
                f"{val_metrics.get('bPQ/Validation', 0):>10.4f} "
                f"{val_metrics.get('mPQ/Validation', 0):>10.4f} "
                f"{lr:>12.6f}"
            )

            if self.early_stopping(fitness):
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        return {"best_bPQ": round(self.best_fitness, 6)}

    @torch.no_grad()
    def test(self, ckpt_path: str | None = None) -> dict:
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})
            print(f"  Loaded checkpoint from {ckpt_path}")

        if self.test_loader is None:
            print("  No test loader provided, skipping test")
            return {}

        test_metrics, fitness = self.validation_epoch(0)
        print(f"\n  Test Results:")
        for k, v in test_metrics.items():
            print(f"    {k}: {v}")
        return test_metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": {k: v.cpu().float() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_fitness": self.best_fitness,
        }
        torch.save(ckpt, self.save_dir / "weights" / "last.pt")
        if is_best:
            torch.save(ckpt, self.save_dir / "weights" / "best.pt")
