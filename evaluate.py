"""Evaluate a trained checkpoint on PanNuke test fold (fold 3).

Supports both baseline LKCell and WaveLKCell models.
Usage:
    python evaluate.py --checkpoint path/to/best.pt --model-type wavellkcell
    python evaluate.py --checkpoint path/to/best.pt --model-type baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.metrics import get_fast_pq, remap_label

TISSUE_TYPES = {
    "Adrenal_gland": 0, "Bile-duct": 1, "Bladder": 2, "Breast": 3,
    "Cervix": 4, "Colon": 5, "Esophagus": 6, "HeadNeck": 7,
    "Kidney": 8, "Liver": 9, "Lung": 10, "Ovarian": 11,
    "Pancreatic": 12, "Prostate": 13, "Skin": 14, "Stomach": 15,
    "Testis": 16, "Thyroid": 17, "Uterus": 18,
}
NUCLEI_TYPES = {
    "Background": 0, "Neoplastic": 1, "Inflammatory": 2,
    "Connective": 3, "Dead": 4, "Epithelial": 5,
}


def build_model(model_type: str, num_nuclei_classes: int, num_tissue_classes: int, device: torch.device):
    if model_type == "wavellkcell":
        from wave_lk_cell.model import WaveLKCell
        model = WaveLKCell(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            pretrained_encoder=False,
        )
    elif model_type == "baseline":
        from wave_lk_cell.baseline.models.cellvit import CellViT
        model = CellViT(
            model256_path="",
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)


def load_checkpoint(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", -1)
        best_fitness = ckpt.get("best_fitness", None)
        print(f"  Checkpoint: epoch={epoch}, best_fitness={best_fitness}")
    else:
        state_dict = ckpt

    state_dict = {k: v.float() for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    return model


def unpack_batch(batch, device, num_nuclei_classes: int):
    imgs = batch[0].to(device)
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
    tissue_indices = torch.tensor(
        [TISSUE_TYPES.get(t, 0) for t in tissue_types],
        dtype=torch.long, device=device,
    )

    gt_nuclei_binary_oh = F.one_hot(masks_dict["nuclei_binary_map"], num_classes=2).float().permute(0, 3, 1, 2).to(device)
    gt_nuclei_type_oh = F.one_hot(masks_dict["nuclei_type_map"], num_classes=num_nuclei_classes).float().permute(0, 3, 1, 2).to(device)
    gt_instance_nuclei = gt_nuclei_type_oh * masks_dict["instance_map"].unsqueeze(1).to(device).int()

    gt = {
        "nuclei_binary_map": gt_nuclei_binary_oh,
        "nuclei_type_map": gt_nuclei_type_oh,
        "hv_map": masks_dict["hv_map"].to(device),
        "instance_map": masks_dict["instance_map"].to(device),
        "instance_types_nuclei": gt_instance_nuclei,
        "tissue_types": tissue_indices,
    }
    return imgs, gt


@torch.no_grad()
def evaluate(model, loader, device, num_nuclei_classes: int, magnification: int, amp: bool):
    model.eval()

    all_dice = []
    all_bpq = []
    all_mpq = []
    all_tissue_correct = 0
    all_tissue_total = 0

    use_amp = amp and device.type == "cuda"

    for batch in tqdm.tqdm(loader, desc="Evaluating"):
        imgs, gt = unpack_batch(batch, device, num_nuclei_classes)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(imgs)

        pred_np = outputs["nuclei_binary_map"].float().softmax(dim=1)
        pred_type = outputs["nuclei_type_map"].float().softmax(dim=1)
        pred_hv = outputs["hv_map"].float()
        tissue_logits = outputs["tissue_types"].float()

        instance_map, instance_types = model.calculate_instance_map(
            {"nuclei_binary_map": pred_np, "nuclei_type_map": pred_type, "hv_map": pred_hv},
            magnification,
        )
        instance_types_nuclei = model.generate_instance_nuclei_map(instance_map, instance_types)

        pred_tissue = torch.argmax(tissue_logits, dim=-1)
        all_tissue_correct += (pred_tissue == gt["tissue_types"]).sum().item()
        all_tissue_total += gt["tissue_types"].shape[0]

        B = pred_np.shape[0]
        for i in range(B):
            pred_binary = torch.argmax(pred_np[i], dim=0)
            gt_binary = torch.argmax(gt["nuclei_binary_map"][i], dim=0).type(torch.uint8)
            intersection = (pred_binary * gt_binary).sum().float()
            union = pred_binary.sum().float() + gt_binary.sum().float()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            all_dice.append(float(dice.cpu()))

            remapped_pred = remap_label(instance_map[i].cpu())
            remapped_gt = remap_label(gt["instance_map"][i].cpu())
            [_, _, pq], _ = get_fast_pq(true=remapped_gt, pred=remapped_pred)
            all_bpq.append(pq)

            per_class_pq = []
            pred_inst_nuclei_np = instance_types_nuclei[i].cpu().numpy().astype(np.int32)
            gt_inst_nuclei_np = gt["instance_types_nuclei"][i].cpu().numpy().astype(np.int32)
            for c in range(num_nuclei_classes):
                pred_c = remap_label(pred_inst_nuclei_np[c])
                gt_c = remap_label(gt_inst_nuclei_np[c])
                if len(np.unique(gt_c)) == 1:
                    per_class_pq.append(np.nan)
                else:
                    [_, _, pq_c], _ = get_fast_pq(pred_c, gt_c, match_iou=0.5)
                    per_class_pq.append(pq_c)
            all_mpq.append(np.nanmean(per_class_pq))

    results = {
        "Dice": float(np.nanmean(all_dice)),
        "bPQ": float(np.nanmean(all_bpq)),
        "mPQ": float(np.nanmean(all_mpq)),
        "Tissue_Acc": all_tissue_correct / max(all_tissue_total, 1),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate WaveLKCell / LKCell baseline on PanNuke test fold")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--model-type", type=str, required=True, choices=["wavellkcell", "baseline"],
                        help="Model architecture to use")
    parser.add_argument("--num-classes", type=int, default=6, help="num_nuclei_classes (default: 6)")
    parser.add_argument("--num-tissue", type=int, default=19, help="num_tissue_classes (default: 19)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--magnification", type=int, default=40)
    parser.add_argument("--amp", action="store_true", help="Enable AMP for inference")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    model = build_model(args.model_type, args.num_classes, args.num_tissue, device)
    model = load_checkpoint(model, args.checkpoint, device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    eval_transforms = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    data = PanNukeData(
        batch_size=args.batch_size,
        test_fold=3,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        eval_transforms=eval_transforms,
    )
    data.setup("test")

    results = evaluate(model, data.test_loader, device, args.num_classes, args.magnification, args.amp)

    print("\n" + "=" * 50)
    print("  TEST RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
