"""Evaluate wavelet stem model on PanNuke fold 3 and output visualization images.

Usage:
    python evaluate_vis.py --checkpoint runs/wavelet_stem/.../weights/best.pt
    python evaluate_vis.py --checkpoint runs/wavelet_stem/.../weights/best.pt --output-dir output_vis
    python evaluate_vis.py --checkpoint runs/wavelet_stem/.../weights/best.pt --max-images 50
    python evaluate_vis.py --checkpoint runs/wavelet_stem/.../weights/best.pt --stratified --max-images 100
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Subset

from wave_lk_cell.baseline.models.cellvit import CellViT, DataclassHVStorage
from wave_lk_cell.baseline.metrics import get_fast_pq, remap_label
from wave_lk_cell.modeling.wavelet.dwt import DWT2
from wave_lk_cell.modeling.wavelet.processors import (
    AdaptivePowerGaborConv,
    ChannelAttention,
    DepthwisePointwiseConv,
    SelfAttention2d,
)
from wave_lk_cell.data.pannuke import PanNukeData

NUCLEI_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]


class LayerNormCF(nn.Module):
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
    def __init__(self, out_channels=96, num_heads=4):
        super().__init__()
        self.dwt1 = DWT2(3)
        self.expand1 = nn.Sequential(
            nn.Conv2d(12, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.dwt2 = DWT2(48)
        self.hh_processor = AdaptivePowerGaborConv(48, 48)
        self.lh_processor = nn.Sequential(SelfAttention2d(48, num_heads=num_heads), nn.BatchNorm2d(48), nn.GELU())
        self.hl_processor = nn.Sequential(SelfAttention2d(48, num_heads=num_heads), nn.BatchNorm2d(48), nn.GELU())
        self.ll_processor = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.GELU())
        self.merge = nn.Sequential(nn.Conv2d(192, out_channels, 1, bias=False), LayerNormCF(out_channels))
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
        bands1 = self.dwt1(x)
        cat1 = torch.cat([bands1["LL"], bands1["LH"], bands1["HL"], bands1["HH"]], dim=1)
        x1 = self.expand1(cat1)
        bands2 = self.dwt2(x1)
        hh2 = self.hh_processor(bands2["HH"])
        lh2 = self.lh_processor(bands2["LH"])
        hl2 = self.hl_processor(bands2["HL"])
        ll2 = self.ll_processor(bands2["LL"])
        cat2 = torch.cat([ll2, lh2, hl2, hh2], dim=1)
        return self.merge(cat2)


def build_model(device):
    model = CellViT(model256_path=None, num_nuclei_classes=6, num_tissue_classes=19, in_channels=3)
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
    return model.to(device)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k: v.float() for k, v in (ckpt.get("model_state_dict", ckpt)).items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded checkpoint from epoch {epoch}")
    return model


def instance_map_to_color(inst_map):
    """Convert instance map (H,W) to RGB color overlay."""
    H, W = inst_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]
    rng = np.random.RandomState(42)
    for idx in inst_ids:
        color = NUCLEI_COLORS[idx % len(NUCLEI_COLORS)]
        mask = inst_map == idx
        color_img[mask] = color
    return color_img


def overlay_instances(image_rgb, inst_map, alpha=0.5):
    """Overlay colored instance map on image."""
    inst_color = instance_map_to_color(inst_map)
    blended = cv2.addWeighted(image_rgb, 1.0, inst_color, alpha, 0)
    contours = np.zeros_like(image_rgb)
    inst_ids = np.unique(inst_map)
    inst_ids = inst_ids[inst_ids > 0]
    for idx in inst_ids:
        mask = (inst_map == idx).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours, cnts, -1, (255, 255, 255), 1)
    blended = cv2.addWeighted(blended, 1.0, contours, 0.8, 0)
    return blended


def build_stratified_indices(dataset, max_images):
    """Sample indices equally across tissue types.

    Returns a list of dataset indices such that each tissue type contributes
    approximately max_images / num_tissues images. If a tissue has fewer
    available images than its quota, the remainder is redistributed.
    """
    tissue_indices = defaultdict(list)
    for i in range(len(dataset)):
        sample = dataset.data[i]
        tissue = sample["tissue"]
        if isinstance(tissue, int):
            tissue = dataset.data.features["tissue"].int2str(tissue)
        tissue_indices[tissue].append(i)

    num_tissues = len(tissue_indices)
    per_tissue = max(max_images // num_tissues, 1)

    selected = []
    remaining_quota = max_images
    tissues_sorted = sorted(tissue_indices.keys())

    for tissue in tissues_sorted:
        indices = tissue_indices[tissue]
        quota = min(per_tissue, len(indices), remaining_quota)
        rng = np.random.RandomState(42)
        chosen = rng.choice(indices, size=quota, replace=False).tolist()
        selected.extend(chosen)
        remaining_quota -= quota

    if remaining_quota > 0:
        all_remaining = []
        for tissue in tissues_sorted:
            chosen_set = set(selected)
            all_remaining.extend(i for i in tissue_indices[tissue] if i not in chosen_set)
        rng = np.random.RandomState(123)
        extra = rng.choice(all_remaining, size=min(remaining_quota, len(all_remaining)), replace=False).tolist()
        selected.extend(extra)

    rng = np.random.RandomState(0)
    rng.shuffle(selected)
    return selected[:max_images]


@torch.no_grad()
def visualize(model, loader, device, output_dir, max_images, magnification, amp):
    model.eval()
    use_amp = amp and device.type == "cuda"
    img_count = 0

    all_dice = []
    all_bpq = []

    for batch in tqdm.tqdm(loader, desc="Evaluating"):
        if img_count >= max_images:
            break

        imgs = batch[0].to(device)
        targets = batch[1]
        B = imgs.shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(imgs)

        pred_np = outputs["nuclei_binary_map"].float().softmax(dim=1)
        pred_type = outputs["nuclei_type_map"].float().softmax(dim=1)
        pred_hv = outputs["hv_map"].float()

        pred_dict = {"nuclei_binary_map": pred_np, "nuclei_type_map": pred_type, "hv_map": pred_hv}
        instance_map, _ = model.calculate_instance_map(pred_dict, magnification)

        for i in range(B):
            if img_count >= max_images:
                break

            img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            img_rgb = (img_np * 255).astype(np.uint8)

            t = targets[i]
            gt_inst = torch.zeros_like(t["binary_map"], dtype=torch.int64)
            masks = t["masks"]
            for j in range(masks.shape[0]):
                gt_inst[masks[j] > 0] = j + 1
            gt_inst_np = gt_inst.numpy()

            pred_inst_np = instance_map[i].cpu().numpy().astype(np.int64)

            pred_binary = torch.argmax(pred_np[i], dim=0).cpu()
            gt_binary = t["binary_map"].long()
            intersection = (pred_binary * gt_binary).sum().float()
            union = pred_binary.sum().float() + gt_binary.sum().float()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            all_dice.append(float(dice))

            remapped_pred = remap_label(pred_inst_np)
            remapped_gt = remap_label(gt_inst_np)
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_pred)
            all_bpq.append(pq)

            tissue_name = t.get("tissue", "unknown")
            if not isinstance(tissue_name, str):
                tissue_name = "unknown"
            folder = output_dir / f"image-{img_count:04d}_{tissue_name}"
            folder.mkdir(exist_ok=True)

            cv2.imwrite(str(folder / "original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            gt_overlay = overlay_instances(img_rgb, gt_inst_np)
            cv2.imwrite(str(folder / "GT-overlay.png"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

            pred_overlay = overlay_instances(img_rgb, pred_inst_np)
            cv2.imwrite(str(folder / "model-output.png"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

            img_count += 1

    print(f"\nSaved {img_count} images to {output_dir}")
    if all_dice:
        print(f"  Avg Dice: {np.nanmean(all_dice):.4f}")
        print(f"  Avg bPQ:  {np.nanmean(all_bpq):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize wavelet stem predictions on PanNuke fold 3")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--magnification", type=int, default=40)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--stratified", action="store_true",
                        help="Sample equally from each tissue type")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Stratified: {args.stratified}")

    model = build_model(device)
    model = load_checkpoint(model, args.checkpoint, device)

    eval_transforms = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    data = PanNukeData(
        batch_size=args.batch_size,
        test_fold=3,
        num_workers=args.num_workers,
        num_classes=6,
        eval_transforms=eval_transforms,
    )
    data.setup("test")

    if args.stratified:
        test_dataset = data.test_loader.dataset
        indices = build_stratified_indices(test_dataset, args.max_images)
        print(f"  Stratified sampling: {len(indices)} images from {len(set(test_dataset.data[i]['tissue'] for i in indices))} tissues")
        subset = Subset(test_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=data.test_loader.collate_fn,
            pin_memory=True,
        )
    else:
        loader = data.test_loader

    visualize(model, loader, device, output_dir, args.max_images, args.magnification, args.amp)


if __name__ == "__main__":
    main()
