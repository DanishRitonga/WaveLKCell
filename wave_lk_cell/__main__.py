from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.losses import (
    DiceLoss,
    FocalTverskyLoss,
    MCFocalTverskyLoss,
    MSGELossMaps,
    MSELossMaps,
    XentropyLoss,
)
from wave_lk_cell.model import WaveLKCell
from wave_lk_cell.trainer import WaveLKCellTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaveLKCell Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=["fit", "test", "fit+test"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


def build_loss_fn_dict(cfg: dict) -> dict:
    loss_registry = {
        "FocalTverskyLoss": FocalTverskyLoss,
        "MCFocalTverskyLoss": lambda **kw: MCFocalTverskyLoss(**kw),
        "xentropy_loss": XentropyLoss,
        "dice_loss": DiceLoss,
        "mse_loss_maps": MSELossMaps,
        "msge_loss_maps": MSGELossMaps,
        "CrossEntropyLoss": nn.CrossEntropyLoss,
    }

    loss_fn_dict = {}
    for branch, branch_losses in cfg.items():
        loss_fn_dict[branch] = {}
        for loss_name, loss_cfg in branch_losses.items():
            fn_name = loss_cfg["loss_fn"]
            weight = loss_cfg.get("weight", 1.0)
            args = loss_cfg.get("args", {})
            if fn_name in ("FocalTverskyLoss", "MCFocalTverskyLoss", "CrossEntropyLoss"):
                loss_fn = loss_registry[fn_name](**args)
            else:
                loss_fn = loss_registry[fn_name]()
            loss_fn_dict[branch][loss_name] = {"loss_fn": loss_fn, "weight": weight}
    return loss_fn_dict


def main() -> None:
    import yaml

    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 19)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    run_name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = str(Path(cfg.get("save_dir", "runs/wavellkcell")) / run_name)

    mode = args.mode or cfg.get("mode", "fit")
    do_fit = "fit" in mode
    do_test = "test" in mode

    data_cfg = cfg["data"]
    batch_size = args.batch_size or data_cfg.get("batch_size", 32)

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

    data = PanNukeData(
        batch_size=batch_size,
        train_fold=data_cfg.get("train_fold", 1),
        val_fold=data_cfg.get("val_fold", 2),
        test_fold=data_cfg.get("test_fold", 3),
        num_workers=data_cfg.get("num_workers", 16),
        num_classes=data_cfg.get("num_classes", 6),
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )

    if do_fit:
        data.setup("fit")
    if do_test:
        data.setup("test")

    model_cfg = cfg["model"]
    model = WaveLKCell(
        num_nuclei_classes=model_cfg.get("num_nuclei_classes", 6),
        num_tissue_classes=model_cfg.get("num_tissue_classes", 19),
        pretrained_encoder=model_cfg.get("pretrained_encoder", True),
    )

    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})
    trainer_cfg = cfg.get("trainer", {})
    loss_cfg = cfg.get("loss", {})
    dataset_config = cfg.get("dataset_config", {})

    loss_fn_dict = build_loss_fn_dict(loss_cfg)

    trainer = WaveLKCellTrainer(
        model=model,
        loss_fn_dict=loss_fn_dict,
        train_loader=data.train_loader if do_fit else None,
        val_loader=data.val_loader if do_fit else None,
        test_loader=data.test_loader if do_test else None,
        lr=float(args.lr or opt_cfg.get("lr", 8e-4)),
        backbone_lr_ratio=float(opt_cfg.get("backbone_lr_ratio", 0.1)),
        betas=tuple(opt_cfg.get("betas", [0.85, 0.95])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
        epochs=int(args.epochs or trainer_cfg.get("epochs", 130)),
        accumulate_grad_batches=int(trainer_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        eta_min=float(sched_cfg.get("eta_min", 1e-5)),
        unfreeze_epoch=int(trainer_cfg.get("unfreeze_epoch", 25)),
        patience=int(trainer_cfg.get("patience", 130)),
        amp=args.amp if args.amp is not None else trainer_cfg.get("amp", True),
        magnification=int(trainer_cfg.get("magnification", 40)),
        num_classes=model_cfg.get("num_nuclei_classes", 6),
        save_dir=save_dir,
        experiment_name=cfg.get("experiment_name", "wavellkcell"),
        dataset_config=dataset_config,
        device=args.device,
    )

    if do_fit:
        print(f"\nTraining: {save_dir}")
        print(f"Params: {sum(p.numel() for p in model.parameters()):,.0f}")
        trainer.fit()

    if do_test:
        ckpt = args.checkpoint or cfg.get("checkpoint")
        best_pt = Path(save_dir) / "weights" / "best.pt"
        if ckpt is None and best_pt.exists():
            ckpt = str(best_pt)
        trainer.test(ckpt_path=ckpt)


if __name__ == "__main__":
    main()
