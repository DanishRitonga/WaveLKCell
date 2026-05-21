"""LKCell baseline training script using HuggingFace PanNuke dataset.

This script uses the original LKCell CellViT model and training pipeline
with our HuggingFace PanNuke data loader as a sanity check.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wave_lk_cell.baseline.base_loss import (
    DiceLoss,
    FocalTverskyLoss,
    MCFocalTverskyLoss,
    MSGELossMaps,
    MSELossMaps,
    XentropyLoss,
)
from wave_lk_cell.baseline.metrics import get_fast_pq, remap_label
from wave_lk_cell.baseline.models.cellvit import CellViT, DataclassHVStorage

from wave_lk_cell.data.datasets.hv_dataset import _compute_hv_map
from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.data.utils.collate_fn import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="LKCell Baseline Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", type=str, default=None)
    return parser.parse_args()


class AverageMeter:
    def __init__(self, name):
        self.name = name
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


def build_loss_fn_dict():
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
            "mcfocaltverskyloss": {"loss_fn": MCFocalTverskyLoss(num_classes=6), "weight": 0.5},
        },
        "tissue_types": {
            "ce": {"loss_fn": nn.CrossEntropyLoss(), "weight": 0.1},
        },
    }


TISSUE_MAP = {
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3,
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7,
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15,
    "Testis": 16, "Thyroid": 17, "Uterus": 18,
}


def unpack_batch(batch, num_classes=6, device="cuda"):
    """Convert our HuggingFace batch format to LKCell's expected format."""
    imgs = batch[0].to(device)
    targets_list = batch[1]

    B = len(targets_list)
    H, W = imgs.shape[2], imgs.shape[3]

    instance_maps = []
    nuclei_binary_maps = []
    nuclei_type_maps = []
    hv_maps = []
    tissue_types = []

    for t in targets_list:
        masks = t["masks"]
        labels = t["labels"]
        binary_map = t["binary_map"]
        hv_map = t["hv_map"]
        type_map = t["type_map"]
        tissue = t.get("tissue", "unknown")
        tissue_idx = t.get("tissue_idx", 0)

        inst_map = torch.zeros_like(binary_map, dtype=torch.int64)
        for j in range(masks.shape[0]):
            inst_map[masks[j] > 0] = j + 1

        instance_maps.append(inst_map)
        nuclei_binary_maps.append(binary_map.long())
        nuclei_type_maps.append(type_map.long())
        hv_maps.append(hv_map)

        if isinstance(tissue, str):
            tissue_types.append(TISSUE_MAP.get(tissue, 0))
        else:
            tissue_types.append(int(tissue_idx))

    masks_dict = {
        "instance_map": torch.stack(instance_maps).to(device),
        "nuclei_binary_map": torch.stack(nuclei_binary_maps).to(device),
        "nuclei_type_map": torch.stack(nuclei_type_maps).to(device),
        "hv_map": torch.stack(hv_maps).to(device),
    }

    return imgs, masks_dict, tissue_types


def unpack_predictions(predictions, model, device, magnification=40):
    """Match LKCell's unpack_predictions exactly."""
    predictions["tissue_types"] = predictions["tissue_types"].to(device)
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)

    (
        predictions["instance_map"],
        predictions["instance_types"],
    ) = model.calculate_instance_map(predictions, magnification)

    predictions["instance_types_nuclei"] = model.generate_instance_nuclei_map(
        predictions["instance_map"], predictions["instance_types"],
    ).to(device)

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
        num_nuclei_classes=6,
    )


def unpack_masks(masks_dict, tissue_types, num_classes=6, device="cuda"):
    """Match LKCell's unpack_masks exactly."""
    gt_nuclei_binary_map_onehot = F.one_hot(
        masks_dict["nuclei_binary_map"], num_classes=2,
    ).float()
    nuclei_type_maps = masks_dict["nuclei_type_map"].long()
    gt_nuclei_type_maps_onehot = F.one_hot(
        nuclei_type_maps, num_classes=num_classes,
    ).float()

    gt = {
        "nuclei_type_map": gt_nuclei_type_maps_onehot.permute(0, 3, 1, 2).to(device),
        "nuclei_binary_map": gt_nuclei_binary_map_onehot.permute(0, 3, 1, 2).to(device),
        "hv_map": masks_dict["hv_map"].to(device),
        "instance_map": masks_dict["instance_map"].to(device),
        "instance_types_nuclei": (
            gt_nuclei_type_maps_onehot * masks_dict["instance_map"][..., None]
        ).permute(0, 3, 1, 2).to(device),
        "tissue_types": torch.tensor(
            tissue_types, dtype=torch.long, device=device,
        ),
    }
    return DataclassHVStorage(
        **gt,
        batch_size=len(tissue_types),
        num_nuclei_classes=num_classes,
    )


def calculate_loss(predictions_dict, gt_dict, loss_fn_dict, device, loss_avg_tracker):
    """Match LKCell's calculate_loss exactly."""
    total_loss = 0

    for branch, pred in predictions_dict.items():
        if branch in ["instance_map", "instance_types", "instance_types_nuclei"]:
            continue
        if branch not in loss_fn_dict:
            continue
        for loss_name, loss_setting in loss_fn_dict[branch].items():
            loss_fn = loss_setting["loss_fn"]
            weight = loss_setting["weight"]
            if loss_name == "msge":
                loss_value = loss_fn(
                    input=pred, target=gt_dict[branch],
                    focus=gt_dict["nuclei_binary_map"], device=device,
                )
            else:
                loss_value = loss_fn(input=pred, target=gt_dict[branch])
            total_loss = total_loss + weight * loss_value
            loss_avg_tracker[f"{branch}_{loss_name}"].update(
                loss_value.detach().cpu().numpy()
            )
    loss_avg_tracker["Total_Loss"].update(total_loss.detach().cpu().numpy())
    return total_loss


def train_epoch(
    model, train_loader, optimizer, scaler, loss_fn_dict, device,
    loss_avg_tracker, mixed_precision, accumulate, epoch,
):
    model.train()
    for key in loss_avg_tracker:
        loss_avg_tracker[key].reset()

    dice_scores = []
    last_opt_step = -1

    loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, batch in loop:
        imgs, masks_dict, tissue_types = unpack_batch(batch, device=device)

        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                predictions_ = model(imgs)

            predictions = unpack_predictions(predictions_, model, device)
            gt = unpack_masks(masks_dict, tissue_types, device=device)

            total_loss = calculate_loss(
                predictions.get_dict(), gt.get_dict(),
                loss_fn_dict, device, loss_avg_tracker,
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if mixed_precision:
                scaler.scale(total_loss / accumulate).backward()
            else:
                (total_loss / accumulate).backward()

            if (batch_idx - last_opt_step) >= accumulate:
                last_opt_step = batch_idx
                if mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        except RuntimeError as e:
            if "out of memory" in str(e):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            raise

        with torch.no_grad():
            pred_dict = predictions.get_dict()
            gt_dict = gt.get_dict()
            for i in range(pred_dict["nuclei_binary_map"].shape[0]):
                pred_binary = torch.argmax(pred_dict["nuclei_binary_map"][i], dim=0)
                gt_binary = torch.argmax(gt_dict["nuclei_binary_map"][i], dim=0).type(torch.uint8)
                intersection = (pred_binary * gt_binary).sum().float()
                union = pred_binary.sum().float() + gt_binary.sum().float()
                dice = (2 * intersection + 1e-8) / (union + 1e-8)
                dice_scores.append(float(dice.cpu()))

        loop.set_postfix({
            "Loss": np.round(loss_avg_tracker["Total_Loss"].avg, 3),
            "Dice": np.round(np.nanmean(dice_scores), 3),
        })

    return {"loss": loss_avg_tracker["Total_Loss"].avg, "dice": np.nanmean(dice_scores)}


@torch.no_grad()
def validate_epoch(
    model, val_loader, loss_fn_dict, device, loss_avg_tracker, mixed_precision, epoch,
):
    model.eval()
    for key in loss_avg_tracker:
        loss_avg_tracker[key].reset()

    dice_scores = []
    pq_scores = []

    loop = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    for batch_idx, batch in loop:
        imgs, masks_dict, tissue_types = unpack_batch(batch, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
            predictions_ = model(imgs)

        predictions = unpack_predictions(predictions_, model, device)
        gt = unpack_masks(masks_dict, tissue_types, device=device)

        _ = calculate_loss(
            predictions.get_dict(), gt.get_dict(),
            loss_fn_dict, device, loss_avg_tracker,
        )

        pred_dict = predictions.get_dict()
        gt_dict = gt.get_dict()

        pred_inst_nuclei = pred_dict["instance_types_nuclei"].cpu().numpy().astype(np.int32)
        gt_inst_nuclei = gt_dict["instance_types_nuclei"].cpu().numpy().astype(np.int32)

        for i in range(pred_dict["nuclei_binary_map"].shape[0]):
            pred_binary = torch.argmax(pred_dict["nuclei_binary_map"][i], dim=0)
            gt_binary = torch.argmax(gt_dict["nuclei_binary_map"][i], dim=0).type(torch.uint8)
            intersection = (pred_binary * gt_binary).sum().float()
            union = pred_binary.sum().float() + gt_binary.sum().float()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores.append(float(dice.cpu()))

            remapped_pred = remap_label(pred_dict["instance_map"][i].cpu())
            remapped_gt = remap_label(gt_dict["instance_map"].cpu()[i])
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_pred)
            pq_scores.append(pq)

        loop.set_postfix({
            "Loss": np.round(loss_avg_tracker["Total_Loss"].avg, 3),
            "Dice": np.round(np.nanmean(dice_scores), 3),
            "bPQ": np.round(np.nanmean(pq_scores), 3),
        })

    return {
        "loss": loss_avg_tracker["Total_Loss"].avg,
        "dice": np.nanmean(dice_scores),
        "bPQ": np.nanmean(pq_scores),
    }


def main():
    args = parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 19)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mixed_precision = args.amp and device.type == "cuda"
    batch_size = int(args.batch_size or cfg.get("data", {}).get("batch_size", 8))
    epochs = int(args.epochs or cfg.get("trainer", {}).get("epochs", 130))
    lr = float(args.lr or cfg.get("optimizer", {}).get("lr", 8e-4))
    accumulate = int(cfg.get("trainer", {}).get("accumulate_grad_batches", 4))

    run_name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"runs/lkcell_baseline/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "weights").mkdir(exist_ok=True)

    train_transforms = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    eval_transforms = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    data = PanNukeData(
        batch_size=batch_size,
        train_fold=cfg.get("data", {}).get("train_fold", 1),
        val_fold=cfg.get("data", {}).get("val_fold", 2),
        test_fold=cfg.get("data", {}).get("test_fold", 3),
        num_workers=cfg.get("data", {}).get("num_workers", 16),
        num_classes=6,
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )
    data.setup("fit")

    print(f"\nLKCell Baseline Training: {save_dir}")
    print(f"Device: {device}, AMP: {mixed_precision}")

    model = CellViT(
        model256_path=None,
        num_nuclei_classes=6,
        num_tissue_classes=19,
        in_channels=3,
    )
    model = model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn_dict = build_loss_fn_dict()
    loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss")}
    for branch, losses in loss_fn_dict.items():
        for ln in losses:
            loss_avg_tracker[f"{branch}_{ln}"] = AverageMeter(f"{branch}_{ln}")

    backbone_params = list(model.parameters())
    other_params = []
    seen = set()
    for p in backbone_params:
        if p not in seen:
            seen.add(p)

    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.85, 0.95), weight_decay=0.05, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    print(f"\n{'Epoch':>5} {'loss':>10} {'Dice':>10} {'bPQ':>10} {'lr':>12}")
    print("-" * 55)

    best_bpq = 0.0
    for epoch in range(epochs):
        train_metrics = train_epoch(
            model, data.train_loader, optimizer, scaler,
            loss_fn_dict, device, loss_avg_tracker, mixed_precision, accumulate, epoch,
        )

        val_metrics = validate_epoch(
            model, data.val_loader, loss_fn_dict, device,
            loss_avg_tracker, mixed_precision, epoch,
        )

        scheduler.step()

        is_best = val_metrics["bPQ"] > best_bpq
        if is_best:
            best_bpq = val_metrics["bPQ"]

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_bpq": best_bpq,
        }
        torch.save(ckpt, save_dir / "weights" / "last.pt")
        if is_best:
            torch.save(ckpt, save_dir / "weights" / "best.pt")

        print(
            f"{epoch+1:>5} {val_metrics['loss']:>10.4f} "
            f"{val_metrics['dice']:>10.4f} {val_metrics['bPQ']:>10.4f} "
            f"{optimizer.param_groups[0]['lr']:>12.6f}"
        )


if __name__ == "__main__":
    main()
