"""Wavelet Stem: LKCell baseline with stem replaced by wavelet-based downsampling.

Replaces the original stem (2x stride-2 convs: 3→48→96 at /4 spatial) with a
2-level Haar DWT that naturally downsamples 256→128→64 while decomposing the
input into frequency bands. Each band is processed by specialized processors
before merging into the 96-channel /4 feature map that feeds Stage 0.

Everything else (stages 0-3, decoder, heads, losses, training loop) is
identical to the LKCell baseline.
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
from wave_lk_cell.modeling.wavelet.dwt import DWT2, IDWT2
from wave_lk_cell.modeling.wavelet.processors import (
    AdaptivePowerGaborConv,
    ChannelAttention,
    DepthwisePointwiseConv,
    SelfAttention2d,
)

from wave_lk_cell.data.datasets.hv_dataset import _compute_hv_map
from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.data.utils.collate_fn import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet Stem Training")
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


class LayerNormCF(nn.Module):
    """LayerNorm in channels-first format (matches UniRepLKNet's LayerNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class WaveletStem(nn.Module):
    """Replace the original stem (2x stride-2 convs) with wavelet downsampling.

    Architecture:
        Input: (B, 3, 256, 256)
        Level 1: DWT2(3) → 4 bands (3,128,128) → concat → Conv(12→48) + BN + GELU
        Level 2: DWT2(48) → 4 bands (48,64,64) → band-specific processors
        Merge:  cat 4 processed bands (192,64,64) → Conv1x1(192→96) + LayerNorm
        Output: (B, 96, 64, 64) — same as original stem

    Band processors at Level 2 (64×64, 48ch):
        HH → AdaptivePowerGaborConv (high-frequency edges, textures)
        LH → SelfAttention2d (horizontal transitions)
        HL → SelfAttention2d (vertical transitions)
        LL → Conv3x3 + BN + GELU (low-frequency base)
    """

    def __init__(self, out_channels=96, num_heads=4):
        super().__init__()

        # Level 1: DWT on raw RGB, then channel expand
        self.dwt1 = DWT2(3)
        self.expand1 = nn.Sequential(
            nn.Conv2d(12, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        # Level 2: DWT on expanded features
        self.dwt2 = DWT2(48)

        # Band-specific processors at 64×64
        self.hh_processor = AdaptivePowerGaborConv(48, 48)
        self.lh_processor = nn.Sequential(
            SelfAttention2d(48, num_heads=num_heads),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.hl_processor = nn.Sequential(
            SelfAttention2d(48, num_heads=num_heads),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.ll_processor = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

        # Merge bands: 4 × 48 = 192 → out_channels
        self.merge = nn.Sequential(
            nn.Conv2d(48 * 4, out_channels, 1, bias=False),
            LayerNormCF(out_channels),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, 256, 256)
        bands1 = self.dwt1(x)
        cat1 = torch.cat([bands1["LL"], bands1["LH"], bands1["HL"], bands1["HH"]], dim=1)
        x1 = self.expand1(cat1)  # (B, 48, 128, 128)

        bands2 = self.dwt2(x1)  # 4 bands at (B, 48, 64, 64)
        hh2 = self.hh_processor(bands2["HH"])
        lh2 = self.lh_processor(bands2["LH"])
        hl2 = self.hl_processor(bands2["HL"])
        ll2 = self.ll_processor(bands2["LL"])

        cat2 = torch.cat([ll2, lh2, hl2, hh2], dim=1)  # (B, 192, 64, 64)
        return self.merge(cat2)  # (B, 96, 64, 64)


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
    imgs = batch[0].to(device)
    targets_list = batch[1]
    instance_maps = []
    nuclei_binary_maps = []
    nuclei_type_maps = []
    hv_maps = []
    tissue_types = []

    for t in targets_list:
        masks = t["masks"]
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
            if mixed_precision:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions_ = model(imgs)
            else:
                predictions_ = model(imgs)

            predictions = unpack_predictions(predictions_, model, device)
            gt = unpack_masks(masks_dict, tissue_types, device=device)

            pred_dict = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in predictions.get_dict().items()}
            gt_dict = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in gt.get_dict().items()}

            total_loss = calculate_loss(
                pred_dict, gt_dict,
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

        if mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predictions_ = model(imgs)
        else:
            predictions_ = model(imgs)

        predictions = unpack_predictions(predictions_, model, device)
        gt = unpack_masks(masks_dict, tissue_types, device=device)

        pred_dict_fp32 = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in predictions.get_dict().items()}
        gt_dict_fp32 = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in gt.get_dict().items()}

        _ = calculate_loss(
            pred_dict_fp32, gt_dict_fp32,
            loss_fn_dict, device, loss_avg_tracker,
        )

        pred_dict = predictions.get_dict()
        gt_dict = gt.get_dict()

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


def create_wavelet_stem_cellvit(device="cuda"):
    """Create CellViT with stem replaced by WaveletStem.

    The WaveletStem replaces downsample_layers[0] (the two stride-2 convs).
    Stages 0-3 remain unchanged, so pretrained weights for stages load directly.
    The encoder forward is patched to use the wavelet stem.
    """
    model = CellViT(
        model256_path=None,
        num_nuclei_classes=6,
        num_tissue_classes=19,
        in_channels=3,
    )

    encoder = model.encoder
    wavelet_stem = WaveletStem(out_channels=96).to(device)
    encoder.wavelet_stem = wavelet_stem

    original_forward = encoder.forward

    def patched_forward(x):
        if encoder.output_mode == 'features':
            outs = []
            input_feature = []
            input_feature.append(encoder.conv(x))
            input_feature.append(encoder.downsample_layers[0][0](x))

            # Wavelet stem replaces downsample_layers[0]
            x = encoder.wavelet_stem(x)

            for stage_idx in range(4):
                if stage_idx > 0:
                    x = encoder.downsample_layers[stage_idx](x)
                x = encoder.stages[stage_idx](x)
                outs.append(encoder.__getattr__(f'norm{stage_idx}')(x))

            logits = encoder.norm(x.mean([-2, -1]))
            logits = encoder.head(logits)
            return logits, outs, input_feature
        else:
            return original_forward(x)

    encoder.forward = patched_forward

    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    wavelet_params = sum(p.numel() for p in wavelet_stem.parameters())
    print(f"Total params: {total:,} (wavelet stem: {wavelet_params:,})")

    return model


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
    save_dir = Path(f"runs/wavelet_stem/{run_name}")
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

    print(f"\nWavelet Stem Training: {save_dir}")
    print(f"Device: {device}, AMP: {mixed_precision}")

    model = create_wavelet_stem_cellvit(device)

    loss_fn_dict = build_loss_fn_dict()
    loss_avg_tracker = {"Total_Loss": AverageMeter("Total_Loss")}
    for branch, losses in loss_fn_dict.items():
        for ln in losses:
            loss_avg_tracker[f"{branch}_{ln}"] = AverageMeter(f"{branch}_{ln}")

    optimizer = AdamW(
        [{"params": list(model.parameters())}],
        lr=lr, betas=(0.85, 0.95), weight_decay=0.05, eps=1e-8,
    )
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
