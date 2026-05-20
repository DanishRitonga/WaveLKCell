"""Diagnostic: run validation with per-sample logging to diagnose bPQ collapse."""
from __future__ import annotations

import argparse
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from wave_lk_cell.data.datasets import TrainingDataset
from wave_lk_cell.data.utils import collate_fn, format_transform
from wave_lk_cell.model import WaveLKCellModel
from wave_lk_cell.post_processing import post_process, _hv_is_smooth, _proc_np_hv_sobel, _proc_np_hv_edt
from wave_lk_cell.metrics.lkcell_metrics import get_fast_pq
import cv2
import albumentations as A


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=None, help="If no checkpoint, simulate N training steps")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = WaveLKCellModel(num_classes=5, num_tissue_classes=19, pretrained_encoder=True).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = {k: v.float().clone() for k, v in ckpt["model_state_dict"].items()}
        model.load_state_dict(state)
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    elif args.epoch:
        print(f"No checkpoint - training {args.epoch} steps first...")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for step in range(args.epoch):
            images = torch.randn(4, 3, 256, 256, device=device)
            targets = []
            for i in range(4):
                masks = torch.zeros(5, 256, 256, device=device)
                bm = torch.zeros(256, 256, device=device)
                tm = torch.zeros(256, 256, dtype=torch.long, device=device)
                hv = torch.zeros(2, 256, 256, device=device)
                labels = torch.zeros(5, dtype=torch.long)
                for j, (cy, cx) in enumerate([(30,30),(80,80),(130,130),(180,50),(200,200)]):
                    masks[j, cy:cy+15, cx:cx+15] = 1.0
                    bm[cy:cy+15, cx:cx+15] = 1
                    tm[cy:cy+15, cx:cx+15] = (j % 4) + 1
                targets.append({
                    'masks': masks, 'labels': labels,
                    'binary_map': bm, 'hv_map': hv, 'type_map': tm,
                    'tissue': torch.tensor(0), 'tissue_idx': 0,
                })
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)
                losses = model.compute_loss(outputs, targets)
            opt.zero_grad()
            scaler.scale(losses['loss']).backward()
            scaler.step(opt)
            scaler.update()
            if step % max(1, args.epoch // 5) == 0:
                print(f"  Step {step}: loss={losses['loss'].item():.4f}")
        print("Training done, running diagnosis...\n")

    model.eval()

    ds = load_dataset("RationAI/PanNuke")
    eval_data = ds["fold2"]
    eval_data.set_transform(format_transform, columns=["image", "instances", "categories"], output_all_columns=True)
    eval_tf = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    dataset = TrainingDataset(eval_data, eval_tf, num_classes=5)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)

    stats = {"sobel": 0, "edt": 0, "zero_inst": 0, "total_pred": 0, "total_gt": 0, "total_tp": 0}
    n = 0

    with torch.no_grad():
        for images, targets in loader:
            if n >= args.max_samples:
                break
            n += 1

            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)

            np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)[:, 1]
            hv_pred = outputs["hv_map"].float()

            for i in range(images.shape[0]):
                np_prob = np_pred[i].cpu().numpy()
                np_binary = (np_prob > 0.5).astype(np.float32)
                hv_np = hv_pred[i].cpu().numpy()
                gt_masks = targets[i]["masks"].cpu().numpy()
                H, W = np_binary.shape

                fg_count = np_binary.sum()
                fg_ratio = fg_count / (H * W)
                np_prob_min = np_prob.min()
                np_prob_max = np_prob.max()
                np_prob_mean = np_prob.mean()

                hv_raw = hv_np.transpose(1, 2, 0)
                hv_min = hv_np.min()
                hv_max = hv_np.max()
                hv_std = hv_np.std()

                is_smooth = _hv_is_smooth(hv_raw[..., 0], hv_raw[..., 1])

                pred = np.stack([np_binary, hv_raw[..., 0], hv_raw[..., 1]], axis=-1)
                if is_smooth:
                    inst = _proc_np_hv_sobel(pred)
                    path = "sobel"
                    if inst.max() == 0:
                        inst = _proc_np_hv_edt(pred)
                        path = "edt_fallback"
                else:
                    inst = _proc_np_hv_edt(pred)
                    path = "edt"

                n_pred = inst.max()
                gt_n = gt_masks.shape[0]

                if is_smooth:
                    stats["sobel"] += 1
                else:
                    stats["edt"] += 1
                if n_pred == 0:
                    stats["zero_inst"] += 1

                gt_inst = np.zeros((H, W), dtype=np.int32)
                for j in range(gt_n):
                    gt_inst[gt_masks[j] > 0] = j + 1

                tp, fp, fn = get_fast_pq(gt_inst, inst, match_iou=0.5) if (n_pred > 0 and gt_n > 0) else (0, 0, gt_n)
                stats["total_pred"] += n_pred
                stats["total_gt"] += gt_n
                stats["total_tp"] += tp

                if n <= 10:
                    print(f"  #{n}: fg={fg_count:.0f} ({fg_ratio:.3f}), "
                          f"np_prob=[{np_prob_min:.3f},{np_prob_max:.3f}] mean={np_prob_mean:.3f}, "
                          f"hv=[{hv_min:.3f},{hv_max:.3f}] std={hv_std:.4f}, "
                          f"smooth={is_smooth}, path={path}, "
                          f"pred={n_pred}, gt={gt_n}, tp={tp}")

    print(f"\n=== Summary ({n} samples) ===")
    print(f"  Sobel: {stats['sobel']}, EDT: {stats['edt']}")
    print(f"  Zero instances: {stats['zero_inst']}/{n}")
    print(f"  Total pred: {stats['total_pred']}, GT: {stats['total_gt']}, TP: {stats['total_tp']}")
    if stats['total_pred'] > 0 or stats['total_gt'] > 0:
        dq = stats['total_tp'] / (stats['total_tp'] + 0.5*(stats['total_pred']-stats['total_tp']) + 0.5*(stats['total_gt']-stats['total_tp']) + 1e-6)
        print(f"  Est bDQ: {dq:.4f}")


if __name__ == "__main__":
    main()
