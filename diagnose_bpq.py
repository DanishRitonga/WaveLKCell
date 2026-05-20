"""Diagnostic script to understand bPQ collapse at epoch 20.

Run this on the training device with a saved checkpoint:
  uv run python diagnose_bpq.py --checkpoint runs/wavellkcell/run3/weights/last.pt
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from wave_lk_cell.data.datasets import TrainingDataset
from wave_lk_cell.data.utils import collate_fn
from wave_lk_cell.metrics.lkcell_metrics import get_fast_pq
from wave_lk_cell.model import WaveLKCellModel
from wave_lk_cell.post_processing import post_process, _hv_is_smooth, _proc_np_hv_sobel, _proc_np_hv_edt
import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = WaveLKCellModel(num_classes=5, num_tissue_classes=19, pretrained_encoder=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k: v.float().clone() for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    ds = load_dataset("RationAI/PanNuke")
    eval_data = ds["fold2"]

    import albumentations as A
    eval_tf = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    from wave_lk_cell.data.datasets import TrainingDataset
    dataset = TrainingDataset(eval_data, eval_tf, num_classes=5)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate_fn)

    total_pred_inst = 0
    total_gt_inst = 0
    total_tp = 0
    sobel_count = 0
    edt_count = 0
    smooth_vals = []
    edge_zero_count = 0
    marker_zero_count = 0
    np_fg_ratios = []
    hv_ranges = []

    n = 0
    with torch.no_grad():
        for images, targets in loader:
            if n >= args.max_samples:
                break
            n += 1

            images = images.to(device)
            targets_dev = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(images)

            np_pred = outputs["nuclei_binary_map"].float().softmax(dim=1)[:, 1]
            hv_pred = outputs["hv_map"].float()

            for i in range(images.shape[0]):
                np_binary = (np_pred[i] > 0.5).cpu().numpy()
                hv_np = hv_pred[i].cpu().numpy()
                gt_masks = targets_dev[i]["masks"].cpu().numpy()
                gt_n = gt_masks.shape[0]

                H, W = np_binary.shape
                fg_ratio = np_binary.sum() / (H * W)
                np_fg_ratios.append(fg_ratio)

                hv_range = max(hv_np.max() - hv_np.min(), abs(hv_np.max()) + abs(hv_np.min()))
                hv_ranges.append(hv_range)

                hv_t = hv_np.transpose(1, 2, 0)
                is_smooth = _hv_is_smooth(hv_t[..., 0], hv_t[..., 1])
                smooth_vals.append(is_smooth)

                pred = np.stack([np_binary, hv_t[..., 0], hv_t[..., 1]], axis=-1)

                if is_smooth:
                    inst_sobel = _proc_np_hv_sobel(pred, object_size=10, ksize=21)
                    n_inst_sobel = inst_sobel.max()
                    if n_inst_sobel > 0:
                        sobel_count += 1
                        inst = inst_sobel
                    else:
                        edge_zero_count += 1
                        inst = _proc_np_hv_edt(pred)
                        edt_count += 1
                else:
                    inst = _proc_np_hv_edt(pred)
                    edt_count += 1

                n_pred = inst.max()

                if n_pred == 0:
                    marker_zero_count += 1

                pred_mask_list = []
                for inst_id in np.unique(inst):
                    if inst_id == 0:
                        continue
                    m = (inst == inst_id).astype(np.float32)
                    pred_mask_list.append(m)

                pred_masks = np.stack(pred_mask_list) if pred_mask_list else np.zeros((0, H, W))
                n_pred_inst = pred_masks.shape[0]

                if n_pred_inst > 0 and gt_n > 0:
                    gt_inst = np.zeros((H, W), dtype=np.int32)
                    for j in range(gt_n):
                        gt_inst[gt_masks[j] > 0] = j + 1
                    tp, _, _ = get_fast_pq(gt_inst, inst, match_iou=0.5)
                else:
                    tp = 0

                total_pred_inst += n_pred_inst
                total_gt_inst += gt_n
                total_tp += tp

                if n <= 5:
                    print(f"  Sample {n}: fg={fg_ratio:.3f}, hv_range={hv_range:.4f}, "
                          f"smooth={is_smooth}, pred={n_pred_inst}, gt={gt_n}, tp={tp}")

    print(f"\n=== Summary ({n} samples) ===")
    print(f"  Sobel path used: {sobel_count}/{n}")
    print(f"  EDT fallback used: {edt_count}/{n}")
    print(f"  Sobel produced 0 markers: {edge_zero_count}")
    print(f"  Total 0 instances: {marker_zero_count}")
    print(f"  Total pred instances: {total_pred_inst}")
    print(f"  Total GT instances: {total_gt_inst}")
    print(f"  Total TP (IoU>0.5): {total_tp}")
    if total_pred_inst > 0:
        dq = total_tp / (total_tp + 0.5 * (total_pred_inst - total_tp) + 0.5 * (total_gt_inst - total_tp) + 1e-6)
        sq = 1.0 if total_tp > 0 else 0.0
        print(f"  Estimated bPQ: {dq * sq:.4f}")
    print(f"  Mean fg ratio: {np.mean(np_fg_ratios):.4f}")
    print(f"  Mean hv range: {np.mean(hv_ranges):.4f}")
    print(f"  Smooth (lap_var<0.1) ratio: {sum(smooth_vals)/len(smooth_vals):.2f}")


if __name__ == "__main__":
    main()
