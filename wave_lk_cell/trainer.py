from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wave_lk_cell.model import NestedMetricCollection, WaveLKCellModel
from wave_lk_cell.metrics import BinaryPanopticQuality, PanopticQuality
from wave_lk_cell.misc.config import save_results


class EarlyStopping:
    def __init__(self, patience: int = 30) -> None:
        self.patience = patience
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float) -> bool:
        if fitness > self.best_fitness or self.best_fitness == 0:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch
        self.possible_stop = delta >= (self.patience - 1)
        return delta >= self.patience


class WaveLKCellTrainer:
    def __init__(
        self,
        model: WaveLKCellModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        lr: float = 8e-4,
        backbone_lr_ratio: float = 0.1,
        betas: tuple[float, float] = (0.85, 0.95),
        weight_decay: float = 0.05,
        epochs: int = 130,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 0.1,
        warmup_epochs: float = 0,
        eta_min: float = 1e-5,
        unfreeze_backbone_at_epoch: int = 25,
        loss_weights: dict[str, float] | None = None,
        save_dir: str = "runs/train",
        experiment_name: str = "wavellkcell",
        patience: int = 30,
        amp: bool = True,
        num_classes: int = 5,
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
        self.warmup_epochs = warmup_epochs
        self.unfreeze_backbone_at_epoch = unfreeze_backbone_at_epoch
        self.loss_weights = loss_weights or {}
        self.num_classes = num_classes
        self.amp = amp and self.device.type == "cuda"

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.save_dir))
        self.stopper = EarlyStopping(patience=patience)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self._freeze_backbone()
        self.optimizer, self.scheduler = self._build_optimizer(
            float(lr), float(backbone_lr_ratio), tuple(betas), float(weight_decay), float(eta_min),
        )
        self.best_fitness = 0.0
        self.start_epoch = 0

        self.val_binary_metrics = NestedMetricCollection(BinaryPanopticQuality)
        self.val_multiclass_metrics = NestedMetricCollection(
            PanopticQuality, num_classes=num_classes,
        )
        self.test_binary_metrics = NestedMetricCollection(BinaryPanopticQuality)
        self.test_multiclass_metrics = NestedMetricCollection(
            PanopticQuality, num_classes=num_classes,
        )

        self.results: list[dict[str, Any]] = []

    def _freeze_backbone(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        if hasattr(self.model.encoder, "gradient_checkpointing_enable"):
            self.model.encoder.gradient_checkpointing_enable()
        torch.cuda.empty_cache()

    def _build_optimizer(
        self, lr: float, backbone_ratio: float, betas: tuple, wd: float, eta_min: float,
    ) -> tuple[AdamW, CosineAnnealingLR]:
        backbone_params = list(self.model.encoder.parameters())
        other_params = [p for p in self.model.parameters() if not any(p is bp for bp in backbone_params)]
        param_groups = [
            {"params": other_params},
            {"params": backbone_params, "lr": lr * backbone_ratio},
        ]
        optimizer = AdamW(param_groups, lr=lr, betas=betas, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=eta_min)
        return optimizer, scheduler

    def _optimizer_step(self, ni: int) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_val)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _apply_warmup(self, ni: int, nw: int, base_lr: float) -> None:
        if nw == 0 or ni > nw:
            return
        for pg in self.optimizer.param_groups:
            pg["lr"] = np.interp(ni, [0, nw], [1e-8, pg["lr"]])

    @torch.no_grad()
    def _validate(self, loader: DataLoader, prefix: str) -> dict[str, float]:
        is_test = prefix == "test"
        metrics = self.test_binary_metrics if is_test else self.val_binary_metrics
        mc_metrics = self.test_multiclass_metrics if is_test else self.val_multiclass_metrics
        metrics.reset()
        mc_metrics.reset()

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=self.amp):
                outputs = self.model(images)
            losses = self.model.compute_loss(
                {k: v.float() for k, v in outputs.items()}, targets, self.loss_weights,
            )

            total_loss += losses["loss"].item()
            n_batches += 1

            self.model.update_binary_metrics(outputs, targets, metrics)
            if is_test:
                self.model.update_metrics(outputs, targets, mc_metrics)

        metrics_dict: dict[str, float] = {}
        binary_results = metrics.compute()
        for tissue, vals in binary_results.items():
            for k, v in vals.items():
                key = f"{prefix}/{tissue}_{k}"
                metrics_dict[key] = round(float(v), 6)

        avg_bpq = float(torch.stack([v["bPQ"] for v in binary_results.values()]).mean()) if binary_results else 0.0
        metrics_dict[f"{prefix}/bPQ"] = round(avg_bpq, 6)

        if is_test:
            mc_results = mc_metrics.compute()
            for tissue, vals in mc_results.items():
                for k, v in vals.items():
                    key = f"{prefix}_mc/{tissue}_{k}"
                    metrics_dict[key] = round(float(v), 6)
            avg_mpq = float(torch.stack([v["mPQ"] for v in mc_results.values()]).mean()) if mc_results else 0.0
            metrics_dict[f"{prefix}/mPQ"] = round(avg_mpq, 6)

        metrics_dict[f"{prefix}/loss"] = round(total_loss / max(n_batches, 1), 6)

        return metrics_dict

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": {k: v.cpu().float() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_fitness": self.best_fitness,
            "results": self.results,
        }
        torch.save(ckpt, self.save_dir / "weights" / "last.pt")
        if is_best:
            torch.save(ckpt, self.save_dir / "weights" / "best.pt")

    def _log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    def fit(self) -> dict[str, float]:
        nb = len(self.train_loader)
        nw = round(self.warmup_epochs * nb) if self.warmup_epochs > 0 else 0
        base_lr = self.optimizer.param_groups[0]["lr"]
        last_opt_step = -1

        print(f"{'Epoch':>5} {'loss':>10} {'np_loss':>10} {'hv_loss':>10} {'type_loss':>10} {'bPQ':>10} {'lr':>12}")
        print("-" * 85)

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            if epoch >= self.unfreeze_backbone_at_epoch and not any(p.requires_grad for p in self.model.encoder.parameters()):
                self._unfreeze_backbone()
                print(f"  [epoch {epoch}] Backbone unfrozen")

            epoch_losses = {"loss": 0.0, "np_loss": 0.0, "hv_loss": 0.0, "type_loss": 0.0}
            n_batches = 0

            for i, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                ni = i + nb * epoch
                self._apply_warmup(ni, nw, base_lr)

                try:
                    with torch.amp.autocast("cuda", enabled=self.amp):
                        outputs = self.model(images)
                        losses = self.model.compute_loss(
                            {k: v.float() for k, v in outputs.items()}, targets, self.loss_weights,
                        )

                    loss_val = losses["loss"]
                    if not torch.isfinite(loss_val):
                        print(f"  [epoch {epoch+1} batch {i}] NaN/Inf loss detected, skipping batch")
                        self.optimizer.zero_grad()
                        continue

                    self.scaler.scale(loss_val).backward()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  [epoch {epoch+1} batch {i}] OOM, skipping batch")
                        self.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    raise

                if ni - last_opt_step >= self.accumulate:
                    self._optimizer_step(ni)
                    last_opt_step = ni

                for k in epoch_losses:
                    epoch_losses[k] += losses[k].item()
                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            lr_dict = {f"lr/pg{idx}": pg["lr"] for idx, pg in enumerate(self.optimizer.param_groups)}
            train_scalars = {f"train/{k}": round(v, 6) for k, v in epoch_losses.items()}
            self._log_scalars(train_scalars, epoch + 1)
            self._log_scalars(lr_dict, epoch + 1)

            val_scalars = {}
            fitness = 0.0
            if self.val_loader is not None:
                torch.cuda.empty_cache()
                val_scalars = self._validate(self.val_loader, "val")
                self._log_scalars(val_scalars, epoch + 1)
                fitness = val_scalars.get("val/bPQ", 0.0)

            is_best = fitness > self.best_fitness or self.best_fitness == 0
            if is_best:
                self.best_fitness = fitness

            self.results.append({
                "epoch": epoch + 1,
                **epoch_losses,
                **{k: round(v, 6) for k, v in val_scalars.items()},
            })

            print(
                f"{epoch+1:>5} {epoch_losses['loss']:>10.4f} {epoch_losses['np_loss']:>10.4f} "
                f"{epoch_losses['hv_loss']:>10.4f} {epoch_losses['type_loss']:>10.4f} "
                f"{fitness:>10.4f} {self.optimizer.param_groups[0]['lr']:>12.6f}"
            )

            self._save_checkpoint(epoch, is_best)
            self.scheduler.step()

            if self.stopper(epoch + 1, fitness):
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        self.writer.flush()
        return {"best_bPQ": round(self.best_fitness, 6)}

    @torch.no_grad()
    def test(self, ckpt_path: str | None = None) -> dict[str, float]:
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
            self.model.load_state_dict(state)
            print(f"  Loaded checkpoint from {ckpt_path}")

        if self.test_loader is None:
            print("  No test loader provided, skipping test")
            return {}

        results = self._validate(self.test_loader, "test")
        self._log_scalars(results, self.epochs)

        save_results(results, self.save_dir, "test_results")
        self.writer.flush()

        print(f"\n  Test Results:")
        for k, v in results.items():
            print(f"    {k}: {v}")

        return results
