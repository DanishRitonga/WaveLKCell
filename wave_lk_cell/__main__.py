from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.misc.config import load_config
from wave_lk_cell.model import WaveLKCellModel
from wave_lk_cell.trainer import WaveLKCellTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveLKCell Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, default=None, choices=["fit", "test", "fit+test"], help="Override mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for test/resume")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g. cuda:0, cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable AMP")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {}
    if args.mode:
        overrides["mode"] = args.mode
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr
    if args.amp is not None:
        overrides["amp"] = args.amp

    seed = cfg.get("seed", 19)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    run_name = args.name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = str(Path(cfg.get("save_dir", "runs/wavellkcell")) / run_name)

    data_cfg = cfg["data"]
    data_cfg.update({k: v for k, v in overrides.items() if k == "batch_size"})

    train_transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    eval_transforms = [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    mode = overrides.get("mode", cfg.get("mode", "fit"))
    do_fit = "fit" in mode
    do_test = "test" in mode or mode == "fit+test"

    data = PanNukeData(
        batch_size=data_cfg.get("batch_size", 16),
        train_fold=data_cfg.get("train_fold", 1),
        val_fold=data_cfg.get("val_fold", 2),
        test_fold=data_cfg.get("test_fold", 3),
        num_workers=data_cfg.get("num_workers", 4),
        num_classes=data_cfg.get("num_classes", 5),
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )

    if do_fit:
        data.setup("fit")
    if do_test and data.test_loader is None:
        data.setup("test")

    model_cfg = cfg["model"]
    model = WaveLKCellModel(
        num_classes=model_cfg.get("num_classes", 5),
        pretrained_encoder=model_cfg.get("pretrained_encoder", True),
    )

    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    trainer_cfg = cfg.get("trainer", {})
    loss_cfg = cfg.get("loss", {})

    trainer = WaveLKCellTrainer(
        model=model,
        train_loader=data.train_loader if do_fit else None,
        val_loader=data.val_loader if do_fit else None,
        test_loader=data.test_loader if do_test else None,
        lr=float(overrides.get("lr", opt_cfg.get("lr", 8e-4))),
        backbone_lr_ratio=float(opt_cfg.get("backbone_lr_ratio", 0.1)),
        betas=tuple(opt_cfg.get("betas", [0.85, 0.95])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
        epochs=int(overrides.get("epochs", trainer_cfg.get("epochs", 130))),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 0.1)),
        warmup_epochs=float(trainer_cfg.get("warmup_epochs", 0)),
        eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        unfreeze_backbone_at_epoch=int(trainer_cfg.get("unfreeze_backbone_at_epoch", 25)),
        loss_weights=loss_cfg,
        save_dir=save_dir,
        experiment_name=cfg.get("experiment_name", "wavellkcell"),
        patience=trainer_cfg.get("patience", 30),
        amp=overrides.get("amp", trainer_cfg.get("amp", True)),
        num_classes=model_cfg.get("num_classes", 5),
        device=args.device,
    )

    if do_fit:
        print(f"\nTraining: {save_dir}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,.2f}M")
        trainer.fit()

    if do_test:
        ckpt = args.checkpoint or cfg.get("checkpoint")
        best_pt = Path(save_dir) / "weights" / "best.pt"
        if ckpt is None and best_pt.exists():
            ckpt = str(best_pt)
        trainer.test(ckpt_path=ckpt)


if __name__ == "__main__":
    main()
