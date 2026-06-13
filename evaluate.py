"""Evaluate a trained checkpoint on PanNuke test fold (fold 3).

Matches RayCastED's LSP-DETR evaluation protocol:
  AJI, AP@0.5, AP@0.7, AP@0.9, AP@0.5:0.05:0.95,
  bPQ, bMPQ, mPQ, mMPQ,
  F1 (centroid, r=12), Precision, Recall,
  Params, FLOPs, Inference Time.

All instance-matching metrics use Hungarian assignment.

Supports both baseline LKCell and WaveLKCell models.
Usage:
    python evaluate.py --checkpoint path/to/best.pt --model-type wavellkcell
    python evaluate.py --checkpoint path/to/best.pt --model-type baseline
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment

from wave_lk_cell.data.pannuke import PanNukeData
from wave_lk_cell.metrics import (
    compute_aji,
    compute_ap,
    compute_centroid_f1,
    compute_pq_masked,
    mask_iou_matrix,
)

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
TISSUE_NAMES = [
    "Adrenal", "BileDuct", "Bladder", "Breast", "Cervix", "Colorectal",
    "Esophagus", "Head&Neck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
]
NUCLEI_NAMES = ['Neoplastic', 'Inflammatory', 'Connective', 'Necrosis', 'Epithelial']


def _patch_wavelet_stage3(model, device, wavelet_mode="residual"):
    import torch.nn as nn
    from wave_lk_cell.modeling.wavelet.wavelet_enhance import MultiWaveletEnhance
    model.encoder.wavelet_enhance = MultiWaveletEnhance(384, mode=wavelet_mode).to(device)
    model.encoder.wavelet_downsample = nn.Sequential(
        nn.Conv2d(384, 768, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(768),
    ).to(device)
    original_forward = model.encoder.forward

    def patched_forward(x):
        if model.encoder.output_mode == 'features':
            outs = []
            input_feature = []
            input_feature.append(model.encoder.conv(x))
            input_feature.append(model.encoder.downsample_layers[0][0](x))
            for stage_idx in range(3):
                x = model.encoder.downsample_layers[stage_idx](x)
                x = model.encoder.stages[stage_idx](x)
                outs.append(model.encoder.__getattr__(f'norm{stage_idx}')(x))
            x = model.encoder.wavelet_enhance(x)
            x = model.encoder.wavelet_downsample(x)
            outs.append(model.encoder.__getattr__(f'norm3')(x))
            logits = model.encoder.norm(x.mean([-2, -1]))
            logits = model.encoder.head(logits)
            return logits, outs, input_feature
        else:
            return original_forward(x)

    model.encoder.forward = patched_forward


def _patch_wavelet_stem(model, device):
    import torch.nn as nn
    from wave_lk_cell.modeling.wavelet.dwt import DWT2
    from wave_lk_cell.modeling.wavelet.processors import (
        AdaptivePowerGaborConv, SelfAttention2d,
    )

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

    wavelet_stem = WaveletStem(out_channels=96).to(device)
    model.encoder.wavelet_stem = wavelet_stem
    original_forward = model.encoder.forward

    def patched_forward(x):
        if model.encoder.output_mode == 'features':
            outs = []
            input_feature = []
            input_feature.append(model.encoder.conv(x))
            input_feature.append(model.encoder.downsample_layers[0][0](x))
            x = model.encoder.wavelet_stem(x)
            for stage_idx in range(4):
                if stage_idx > 0:
                    x = model.encoder.downsample_layers[stage_idx](x)
                x = model.encoder.stages[stage_idx](x)
                outs.append(model.encoder.__getattr__(f'norm{stage_idx}')(x))
            logits = model.encoder.norm(x.mean([-2, -1]))
            logits = model.encoder.head(logits)
            return logits, outs, input_feature
        else:
            return original_forward(x)

    model.encoder.forward = patched_forward


def build_model(model_type: str, num_nuclei_classes: int, num_tissue_classes: int, device: torch.device, wavelet: bool = False, wavelet_mode: str = "residual"):
    if model_type == "wavellkcell":
        from wave_lk_cell.model import WaveLKCell
        model = WaveLKCell(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            pretrained_encoder=False,
        )
    elif model_type in ("baseline", "wavelet-stage3", "wavelet-stem"):
        from wave_lk_cell.baseline.models.cellvit import CellViT
        model = CellViT(
            model256_path="",
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
        )
        if model_type == "wavelet-stage3" or (model_type == "baseline" and wavelet):
            _patch_wavelet_stage3(model, device, wavelet_mode)
        elif model_type == "wavelet-stem":
            _patch_wavelet_stem(model, device)
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

def _polygon_area(poly):
    """Shoelace area for recall diagnosis (uses centroid + rays format)."""
    cx, cy = poly[0], poly[1]
    rays = poly[2:]
    if len(rays) < 3:
        return 0.0
    angles = np.linspace(0, 2 * np.pi, len(rays), endpoint=False)
    vx = cx + rays * np.cos(angles)
    vy = cy + rays * np.sin(angles)
    return 0.5 * abs(np.dot(vx, np.roll(vy, 1)) - np.dot(vy, np.roll(vx, 1)))


def _extract_instance_masks(inst_map, inst_ids):
    """Extract per-instance binary masks from an instance map."""
    masks = []
    for inst_id in inst_ids:
        m = (inst_map == inst_id).astype(np.uint8)
        if m.sum() > 0:
            masks.append(m)
    return masks


def _get_gt_data(gt_inst_map, gt_inst_types, num_classes):
    gt_ids = sorted(set(np.unique(gt_inst_map)) - {0})
    masks = []
    classes = []
    centroids = []
    for inst_id in gt_ids:
        m = (gt_inst_map == inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        masks.append(m)
        cls_i = 0
        for c in range(num_classes):
            if (gt_inst_types[c] == inst_id).any():
                cls_i = c
                break
        classes.append(cls_i)
        com = center_of_mass(m)
        centroids.append([com[1], com[0]])
    return masks, np.array(classes, dtype=int) if classes else np.array([], dtype=int), np.array(centroids) if centroids else np.zeros((0, 2))


def _get_pred_data(inst_pred, type_pred):
    pred_ids = sorted(type_pred.keys())
    masks = []
    confs = []
    classes = []
    centroids = []
    for inst_id in pred_ids:
        m = (inst_pred == inst_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        masks.append(m)
        confs.append(type_pred[inst_id].get('type_prob', 0.0))
        classes.append(type_pred[inst_id]['type'])
        centroids.append(type_pred[inst_id]['centroid'])
    return masks, np.array(confs), np.array(classes, dtype=int) if classes else np.array([], dtype=int), np.array(centroids) if centroids else np.zeros((0, 2))

def _print_tissue_breakdown(results, metrics):
    panuke_tissues = [
        "Adrenal", "BileDuct", "Bladder", "Breast", "Cervix", "Colorectal",
        "Esophagus", "Head&Neck", "Kidney", "Liver", "Lung", "Ovarian",
        "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
    ]
    n_groups = len(panuke_tissues)
    tp = [0] * n_groups; fp = [0] * n_groups; fn = [0] * n_groups
    n_gt_g = [0] * n_groups; seen = [set() for _ in range(n_groups)]

    for ri, r in enumerate(results):
        g = int(r.get('tissue', 0))
        if g >= n_groups:
            continue
        seen[g].add(ri)
        gt_r = r['gt_centroids']; pred_r = r['pred_centroids']
        n_gt_g[g] += len(gt_r)
        if len(gt_r) == 0:
            fp[g] += len(pred_r); continue
        if len(pred_r) == 0:
            fn[g] += len(gt_r); continue
        dist = np.linalg.norm(gt_r[:, :2][:, None] - pred_r[:, :2][None, :], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        t = int((dist[ri_, ci_] <= 12).sum())
        tp[g] += t; fp[g] += len(pred_r) - t; fn[g] += len(gt_r) - t

    t_aji = metrics.get('tissue_aji', {})
    t_bpq = metrics.get('tissue_bpq', {})
    t_mpq = metrics.get('tissue_mpq', {})
    has_mask = bool(t_aji or t_bpq or t_mpq)

    hdr = f'  {"Group":<14} {"Imgs":>5} {"Prec":>7} {"Recall":>7} {"F1":>7}'
    sep = f'  {"-" * 14} {"-" * 5} {"-" * 7} {"-" * 7} {"-" * 7}'
    fmt = f'  {{name:<14}} {{ni:>5}} {{prec:>7.4f}} {{rec:>7.4f}} {{f1:>7.4f}}'
    if has_mask:
        hdr += f' {"AJI":>7} {"bPQ":>7} {"mPQ":>7}'
        sep += f' {"-" * 7} {"-" * 7} {"-" * 7}'
        fmt += f' {{aji:>7.4f}} {{bpq:>7.4f}} {{mpq:>7.4f}}'

    w = 85 if has_mask else 60
    bar = '=' * w
    print(f'\n{bar}')
    print(f'  Tissue Type Breakdown')
    print(bar)
    print(hdr)
    print(sep)

    for g in range(n_groups):
        if n_gt_g[g] == 0:
            continue
        prec = tp[g] / (tp[g] + fp[g]) if (tp[g] + fp[g]) else 0
        rec = tp[g] / (tp[g] + fn[g]) if (tp[g] + fn[g]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        vals = [panuke_tissues[g], len(seen[g]), prec, rec, f1]
        if has_mask:
            aji_arr = np.array(t_aji.get(g, []))
            bpq_arr = np.array(t_bpq.get(g, []))
            aji_m = float(np.mean(aji_arr)) if len(aji_arr) else 0.0
            bpq_m = float(np.mean(bpq_arr)) if len(bpq_arr) else 0.0
            mpq_vals = [np.mean([x for x in v if x > 0]) for v in t_mpq.get(g, {}).values() if v and any(x > 0 for x in v)]
            mpq_m = float(np.mean(mpq_vals)) if mpq_vals else 0.0
            vals += [aji_m, bpq_m, mpq_m]
        print(fmt.format(name=vals[0], ni=vals[1], prec=vals[2], rec=vals[3], f1=vals[4], **({'aji': vals[5], 'bpq': vals[6], 'mpq': vals[7]} if has_mask else {})))

    tot_tp = sum(tp); tot_fp = sum(fp); tot_fn = sum(fn)
    p_t = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0
    r_t = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0
    f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) else 0
    total_vals = ['TOTAL', len(results), p_t, r_t, f_t]
    if has_mask:
        total_vals += [metrics['aji'], metrics['bpq'], metrics['mpq']]
    print(fmt.format(name=total_vals[0], ni=total_vals[1], prec=total_vals[2], rec=total_vals[3], f1=total_vals[4], **({'aji': total_vals[5], 'bpq': total_vals[6], 'mpq': total_vals[7]} if has_mask else {})))

    if has_mask:
        per_t_aji = [float(np.mean(t_aji.get(g))) for g in range(n_groups) if t_aji.get(g)]
        per_t_bpq = [float(np.mean(t_bpq.get(g))) for g in range(n_groups) if t_bpq.get(g)]
        per_t_mpq = []
        for g in range(n_groups):
            v = t_mpq.get(g, {})
            if v:
                vp = [np.mean([x for x in lst if x > 0]) for lst in v.values() if lst and any(x > 0 for x in lst)]
                if vp:
                    per_t_mpq.append(float(np.mean(vp)))
        am = np.mean(per_t_aji) if per_t_aji else 0; as_ = np.std(per_t_aji) if len(per_t_aji) > 1 else 0
        bm = np.mean(per_t_bpq) if per_t_bpq else 0; bs_ = np.std(per_t_bpq) if len(per_t_bpq) > 1 else 0
        mm = np.mean(per_t_mpq) if per_t_mpq else 0; ms_ = np.std(per_t_mpq) if len(per_t_mpq) > 1 else 0
        print(f'  {"Avg":<14} {"":>5} {"":>7} {"":>7} {"":>7} {am:>7.4f} {bm:>7.4f} {mm:>7.4f}')
        print(f'  {"Std":<14} {"":>5} {"":>7} {"":>7} {"":>7} {as_:>7.4f} {bs_:>7.4f} {ms_:>7.4f}')
    print(bar)


def _print_nuclei_breakdown(results, num_classes):
    names = ['Neoplastic', 'Inflammatory', 'Connective', 'Necrosis', 'Epithelial']
    class_gt = [0] * num_classes
    class_pred = [0] * num_classes
    class_tp = [0] * num_classes

    for r in results:
        gt_c = r['gt_cls']; pred_c = r['pred_cls']
        gt_p = r['gt_centroids']; pred_p = r['pred_centroids']
        for c in gt_c:
            if int(c) < num_classes:
                class_gt[int(c)] += 1
        for c in pred_c:
            if int(c) < num_classes:
                class_pred[int(c)] += 1

        n_gt = len(gt_p); n_pred = len(pred_p)
        if n_gt == 0 or n_pred == 0:
            continue
        dist = np.linalg.norm(gt_p[:, :2][:, None] - pred_p[:, :2][None, :], axis=2)
        ri_, ci_ = linear_sum_assignment(dist)
        for ri, ci in zip(ri_, ci_):
            if dist[ri, ci] <= 12:
                if gt_c[ri] == pred_c[ci]:
                    class_tp[int(gt_c[ri])] += 1

    print(f'\n{"=" * 70}')
    print(f'  Nuclei Class Breakdown (Centroid F1, class-matched)')
    print(f'{"=" * 70}')
    print(f'  {"Class":<14} {"Prec":>7} {"Recall":>7} {"F1":>7}')
    print(f'  {"-" * 14} {"-" * 7} {"-" * 7} {"-" * 7}')
    for c in range(num_classes):
        name = names[c] if c < len(names) else f'cls_{c}'
        prec = class_tp[c] / class_pred[c] if class_pred[c] else 0
        rec = class_tp[c] / class_gt[c] if class_gt[c] else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        print(f'  {name:<14} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}')
    print(f'{"=" * 70}')

def _print_recall_diagnosis(results, num_classes):
    """Break down recall by GT size bin, class, and nearest-prediction distance."""
    names = ['Neoplastic', 'Inflammatory', 'Connective', 'Necrosis', 'Epithelial']

    gt_areas_matched = []
    gt_areas_unmatched = []
    class_total = [0] * num_classes
    class_matched = [0] * num_classes
    unmatched_dists = []
    unmatched_nearest_conf = []
    unmatched_nearest_cls = []

    for r in results:
        gt_polys = r['gt_centroids']
        pred_polys = r['pred_centroids']
        pred_confs = r['pred_confs']
        pred_cls = r['pred_cls']
        gt_cls = r['gt_cls']

        n_gt = len(gt_polys)
        n_pred = len(pred_polys)

        if n_gt == 0:
            continue

        gt_areas = np.array([_polygon_area(p) if hasattr(p, '__len__') and len(p) > 2 else 0 for p in r.get('gt_polys_full', gt_polys)])
        if len(gt_areas) != n_gt:
            gt_areas = np.full(n_gt, 100.0)

        if n_pred > 0:
            pred_cxcy = pred_polys[:, :2] if pred_polys.ndim == 2 else pred_polys.reshape(-1, 2)
            gt_cxcy = gt_polys[:, :2] if gt_polys.ndim == 2 else gt_polys.reshape(-1, 2)
            dist_matrix = np.linalg.norm(gt_cxcy[:, None, :] - pred_cxcy[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            matched = np.zeros(n_gt, dtype=bool)
            matched_dists = dist_matrix[row_ind, col_ind]
            for ri, ci, d in zip(row_ind, col_ind, matched_dists):
                if d <= 12.0:
                    matched[ri] = True
        else:
            matched = np.zeros(n_gt, dtype=bool)
            pred_cxcy = np.zeros((0, 2))

        for j in range(n_gt):
            cls_id = int(gt_cls[j]) if j < len(gt_cls) else 0
            if cls_id < num_classes:
                class_total[cls_id] += 1
                if matched[j]:
                    class_matched[cls_id] += 1

            if matched[j]:
                gt_areas_matched.append(float(gt_areas[j]))
            else:
                gt_areas_unmatched.append(float(gt_areas[j]))
                if n_pred > 0:
                    dists_j = np.linalg.norm(pred_cxcy - gt_polys[j, :2], axis=1)
                    nearest_idx = int(np.argmin(dists_j))
                    unmatched_dists.append(float(dists_j[nearest_idx]))
                    unmatched_nearest_conf.append(float(pred_confs[nearest_idx]) if len(pred_confs) > nearest_idx else 0.0)
                    unmatched_nearest_cls.append(int(pred_cls[nearest_idx]) if len(pred_cls) > nearest_idx else -1)
                else:
                    unmatched_dists.append(float('inf'))
                    unmatched_nearest_conf.append(0.0)
                    unmatched_nearest_cls.append(-1)

    total_gt = sum(class_total)
    total_matched = sum(class_matched)
    recall = total_matched / total_gt if total_gt > 0 else 0.0

    print('\n' + '=' * 65)
    print('Recall Diagnosis')
    print('=' * 65)
    print(f'  Overall: {total_matched}/{total_gt} = {recall:.4f}')

    print(f'\n  {"Class":<18} {"Total":>8} {"Matched":>8} {"Recall":>8}')
    print(f'  {"-" * 18} {"-" * 8} {"-" * 8} {"-" * 8}')
    for c in range(num_classes):
        if class_total[c] > 0:
            r = class_matched[c] / class_total[c]
            n = names[c] if c < len(names) else f'class_{c}'
            print(f'  {n:<18} {class_total[c]:>8} {class_matched[c]:>8} {r:>8.4f}')

    if gt_areas_matched or gt_areas_unmatched:
        all_areas = np.array(gt_areas_matched + gt_areas_unmatched)
        if len(all_areas) > 0:
            p33 = np.percentile(all_areas, 33)
            p67 = np.percentile(all_areas, 67)
            bins = [
                ('Small (<P33)', lambda a: a < p33),
                ('Medium (P33-P67)', lambda a: (a >= p33) & (a < p67)),
                ('Large (>P67)', lambda a: a >= p67),
            ]
            print(f'\n  {"Size Bin":<20} {"Area Range":<18} {"Total":>8} {"Matched":>8} {"Recall":>8}')
            print(f'  {"-" * 20} {"-" * 18} {"-" * 8} {"-" * 8} {"-" * 8}')
            for label, cond in bins:
                t = sum(1 for a in gt_areas_matched + gt_areas_unmatched if cond(a))
                m = sum(1 for a in gt_areas_matched if cond(a))
                min_a = min((a for a in all_areas if cond(a)), default=0)
                max_a = max((a for a in all_areas if cond(a)), default=0)
                rng = f'{min_a:.0f}-{max_a:.0f}px^2'
                rec = m / t if t > 0 else 0.0
                print(f'  {label:<20} {rng:<18} {t:>8} {m:>8} {rec:>8.4f}')

    if unmatched_dists:
        ud = np.array(unmatched_dists)
        finite = ud[np.isfinite(ud)]
        print(f'\n  Unmatched GT — Distance to nearest prediction:')
        print(f'    <5px (near miss): {int(np.sum(finite < 5))}/{len(unmatched_dists)} ({100 * sum(finite < 5) / len(unmatched_dists):.1f}%)')
        print(f'    5-12px (drifted):  {int(np.sum((finite >= 5) & (finite <= 12)))}/{len(unmatched_dists)} ({100 * sum((finite >= 5) & (finite <= 12)) / len(unmatched_dists):.1f}%)')
        print(f'    >12px (truly miss):{int(np.sum(finite > 12))}/{len(unmatched_dists)} ({100 * sum(finite > 12) / len(unmatched_dists):.1f}%)')
        no_det = int(np.isinf(ud).sum())
        if no_det > 0:
            print(f'    No predictions:    {no_det}/{len(unmatched_dists)} ({100 * no_det / len(unmatched_dists):.1f}%)')

    print('=' * 65)

@torch.no_grad()
def evaluate(model, loader, device, num_nuclei_classes, magnification, amp):
    model.eval()
    use_amp = amp and device.type == "cuda"

    iou_thresholds = sorted(set(round(x, 2) for x in np.arange(0.5, 1.0, 0.05)))

    all_dice = []
    aji_scores = []
    bpq_scores = []
    bmpq_scores = []
    class_pq = {c: [] for c in range(num_nuclei_classes)}
    class_mpq = {c: [] for c in range(num_nuclei_classes)}
    centroid_tp = 0; centroid_fp = 0; centroid_fn = 0
    tissue_correct = 0; tissue_total = 0

    class_set = set()
    for c in range(num_nuclei_classes):
        class_set.add(c)

    ap_stats = {}
    for cls_id in sorted(class_set):
        ap_stats[cls_id] = {t: {'tp': [], 'fp': [], 'conf': [], 'n_gt': 0} for t in iou_thresholds}

    tissue_aji = {t: [] for t in range(19)}
    tissue_bpq = {t: [] for t in range(19)}
    tissue_mpq = {t: {c: [] for c in range(num_nuclei_classes)} for t in range(19)}

    results_for_diag = []

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="Evaluating")):
        imgs, gt = unpack_batch(batch, device, num_nuclei_classes)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = model(imgs)

        pred_np = outputs["nuclei_binary_map"].float().softmax(dim=1)
        pred_type = outputs["nuclei_type_map"].float().softmax(dim=1)
        pred_hv = outputs["hv_map"].float()
        tissue_logits = outputs["tissue_types"].float()

        instance_map, type_preds = model.calculate_instance_map(
            {"nuclei_binary_map": pred_np, "nuclei_type_map": pred_type, "hv_map": pred_hv},
            magnification,
        )
        instance_types_nuclei = model.generate_instance_nuclei_map(instance_map, type_preds)
        if instance_types_nuclei.dim() == 4 and instance_types_nuclei.shape[-1] == num_nuclei_classes:
            instance_types_nuclei = instance_types_nuclei.permute(0, 3, 1, 2)

        pred_tissue = torch.argmax(tissue_logits, dim=-1)
        tissue_correct += (pred_tissue == gt["tissue_types"]).sum().item()
        tissue_total += gt["tissue_types"].shape[0]

        B = pred_np.shape[0]
        for i in range(B):
            inst_pred = instance_map[i].cpu().numpy().astype(np.int32)
            type_pred = type_preds[i]
            pred_masks, pred_confs, pred_cls, pred_centroids = _get_pred_data(inst_pred, type_pred)
            n_pred = len(pred_masks)

            gt_inst_map = gt["instance_map"][i].cpu().numpy().astype(np.int32)
            gt_inst_types = gt["instance_types_nuclei"][i].cpu().numpy().astype(np.int32)
            gt_masks, gt_cls, gt_centroids = _get_gt_data(gt_inst_map, gt_inst_types, num_nuclei_classes)
            n_gt = len(gt_masks)

            results_for_diag.append({
                'pred_centroids': pred_centroids,
                'pred_confs': pred_confs,
                'pred_cls': pred_cls,
                'gt_centroids': gt_centroids,
                'gt_cls': gt_cls,
                'tissue': gt["tissue_types"][i].item(),
            })

            # Dice
            pred_binary = torch.argmax(pred_np[i], dim=0).cpu()
            gt_binary = torch.argmax(gt["nuclei_binary_map"][i], dim=0).cpu().type(torch.uint8)
            intersection = (pred_binary * gt_binary).sum().float()
            union = pred_binary.sum().float() + gt_binary.sum().float()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            all_dice.append(float(dice))

            # AJI
            aji = compute_aji(pred_masks, gt_masks)
            aji_scores.append(aji)

            # bPQ / bMPQ
            if pred_masks:
                pred_bin_mask = np.stack(pred_masks).max(axis=0).astype(np.uint8)
            else:
                pred_bin_mask = np.zeros((inst_pred.shape[0], inst_pred.shape[1]), dtype=np.uint8)
            if gt_masks:
                gt_bin_mask = np.stack(gt_masks).max(axis=0).astype(np.uint8)
            else:
                gt_bin_mask = np.zeros((inst_pred.shape[0], inst_pred.shape[1]), dtype=np.uint8)
            bpq, _, _ = compute_pq_masked([pred_bin_mask], [gt_bin_mask])
            bpq_scores.append(bpq)
            if gt_bin_mask.sum() > 0:
                bmpq, _, _ = compute_pq_masked([pred_bin_mask], [gt_bin_mask], mask=gt_bin_mask > 0)
            else:
                bmpq = 0.0
            bmpq_scores.append(bmpq)

            # Per-class mPQ / mMPQ
            gt_any = gt_bin_mask > 0
            class_pq_img = {}
            for cls_id in range(num_nuclei_classes):
                pred_idx = [j for j, c in enumerate(pred_cls) if c == cls_id]
                gt_idx = [j for j, c in enumerate(gt_cls) if c == cls_id]
                pred_cls_masks = [pred_masks[j] for j in pred_idx]
                gt_cls_masks = [gt_masks[j] for j in gt_idx]
                pq_c, _, _ = compute_pq_masked(pred_cls_masks, gt_cls_masks)
                class_pq[cls_id].append(pq_c)
                class_pq_img[cls_id] = pq_c
                if gt_any.sum() > 0:
                    mpq_c, _, _ = compute_pq_masked(pred_cls_masks, gt_cls_masks, mask=gt_any)
                else:
                    mpq_c = 0.0
                class_mpq[cls_id].append(mpq_c)

            # AP per-class per-threshold
            for cls_id in sorted(class_set):
                pred_idx = [j for j, c in enumerate(pred_cls) if c == cls_id]
                gt_idx = [j for j, c in enumerate(gt_cls) if c == cls_id]
                cls_pred_masks = [pred_masks[j] for j in pred_idx]
                cls_gt_masks = [gt_masks[j] for j in gt_idx]
                cls_confs = pred_confs[np.isin(pred_cls, [cls_id])]

                n_pred_cls = len(cls_pred_masks)
                n_gt_cls = len(cls_gt_masks)

                for t in iou_thresholds:
                    ap_stats[cls_id][t]['n_gt'] += n_gt_cls

                if n_pred_cls > 0 and n_gt_cls > 0:
                    iou_mat = mask_iou_matrix(cls_pred_masks, cls_gt_masks)
                    row_ind, col_ind = linear_sum_assignment(-iou_mat)
                    matched_iou = iou_mat[row_ind, col_ind]
                else:
                    row_ind = np.array([], dtype=int)
                    matched_iou = np.array([], dtype=float)

                for t in iou_thresholds:
                    if n_gt_cls > 0:
                        valid = matched_iou >= t
                        matched_pred = set(row_ind[valid].tolist())
                    else:
                        matched_pred = set()

                    for pi in range(n_pred_cls):
                        is_tp = pi in matched_pred
                        ap_stats[cls_id][t]['conf'].append(float(cls_confs[pi]) if pi < len(cls_confs) else 0.0)
                        ap_stats[cls_id][t]['tp'].append(is_tp)
                        ap_stats[cls_id][t]['fp'].append(not is_tp)

            # Centroid F1
            tp_i, fp_i, fn_i = compute_centroid_f1(pred_centroids, gt_centroids)
            centroid_tp += tp_i; centroid_fp += fp_i; centroid_fn += fn_i

            # Tissue tracking
            tissue = gt["tissue_types"][i].item()
            if tissue < 19 and aji > -1:
                tissue_aji[tissue].append(aji)
                tissue_bpq[tissue].append(bpq)
                for cls_id in range(num_nuclei_classes):
                    if class_pq_img.get(cls_id, 0) > 0:
                        tissue_mpq[tissue][cls_id].append(class_pq_img[cls_id])

    # --- Aggregate ---
    mean_aji = np.mean(aji_scores)
    mean_bpq = np.mean(bpq_scores)
    mean_bmpq = np.mean(bmpq_scores)

    mpq_values = []
    mmpq_values = []
    for c in range(num_nuclei_classes):
        valid_pq = [v for v in class_pq[c] if v > 0]
        valid_mpq = [v for v in class_mpq[c] if v > 0]
        if valid_pq:
            mpq_values.append(np.mean(valid_pq))
        if valid_mpq:
            mmpq_values.append(np.mean(valid_mpq))
    mean_mpq = np.mean(mpq_values) if mpq_values else 0.0
    mean_mmpq = np.mean(mmpq_values) if mmpq_values else 0.0

    # AP
    ap_results = {}
    for t in iou_thresholds:
        aps = []
        for cls_id in sorted(class_set):
            stats = ap_stats[cls_id][t]
            n_gt = stats['n_gt']
            if n_gt == 0:
                continue
            confs = np.array(stats['conf'])
            tps = np.array(stats['tp'])
            fps = np.array(stats['fp'])
            if len(confs) == 0:
                aps.append(0.0)
                continue
            order = np.argsort(-confs)
            tps = tps[order]
            fps = fps[order]
            cum_tp = np.cumsum(tps)
            cum_fp = np.cumsum(fps)
            precision = cum_tp / (cum_tp + cum_fp)
            recall = cum_tp / n_gt
            aps.append(compute_ap(recall, precision))
        ap_results[t] = {'AP': np.mean(aps) if aps else 0.0}

    prec = centroid_tp / (centroid_tp + centroid_fp) if (centroid_tp + centroid_fp) > 0 else 0.0
    rec = centroid_tp / (centroid_tp + centroid_fn) if (centroid_tp + centroid_fn) > 0 else 0.0
    f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'dice': float(np.mean(all_dice)),
        'tissue_acc': tissue_correct / max(tissue_total, 1),
        'aji': mean_aji,
        'bpq': mean_bpq,
        'bmpq': mean_bmpq,
        'mpq': mean_mpq,
        'mmpq': mean_mmpq,
        'ap': ap_results,
        'centroid': {'precision': prec, 'recall': rec, 'f1': f1_c},
        'tissue_aji': tissue_aji,
        'tissue_bpq': tissue_bpq,
        'tissue_mpq': tissue_mpq,
    }, results_for_diag


def benchmark_inference(model, loader, device, n_warmup=10):
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['img'].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= n_warmup:
                times.append(elapsed / imgs.shape[0])
    return np.mean(times) if times else 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate WaveLKCell / LKCell baseline on PanNuke test fold (LSP-DETR protocol)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["wavellkcell", "baseline", "wavelet-stage3", "wavelet-stem"],
                        help="Model architecture to use")
    parser.add_argument("--num-classes", type=int, default=6, help="num_nuclei_classes (default: 6)")
    parser.add_argument("--num-tissue", type=int, default=19, help="num_tissue_classes (default: 19)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--magnification", type=int, default=40)
    parser.add_argument("--amp", action="store_true", help="Enable AMP for inference")
    parser.add_argument("--wavelet", action="store_true", help="Baseline has wavelet Stage3 replacement")
    parser.add_argument("--wavelet-mode", type=str, default="residual",
                        choices=["residual", "no_residual", "gated"],
                        help="Wavelet enhance mode (for wavelet-stage3 only)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    model = build_model(args.model_type, args.num_classes, args.num_tissue, device, wavelet=args.wavelet, wavelet_mode=args.wavelet_mode)
    model = load_checkpoint(model, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} ({n_params / 1e6:.2f}M)")

    eval_transforms = [A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    data = PanNukeData(
        batch_size=args.batch_size,
        test_fold=3,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        eval_transforms=eval_transforms,
    )
    data.setup("test")

    metrics, diag_results = evaluate(model, data.test_loader, device, args.num_classes, args.magnification, args.amp)

    ap_res = metrics['ap']
    ap50 = ap_res.get(0.5, {}).get('AP', 0.0)
    ap70 = ap_res.get(0.7, {}).get('AP', 0.0)
    ap90 = ap_res.get(0.9, {}).get('AP', 0.0)
    ap50_95 = np.mean([ap_res[t]['AP'] for t in sorted(ap_res.keys())])
    f12 = metrics['centroid']

    # --- Inference time ---
    inf_dataset = data.test_loader.dataset
    from torch.utils.data import DataLoader
    inf_dl = DataLoader(inf_dataset, batch_size=1, shuffle=False, num_workers=0)
    print("Benchmarking inference time...", flush=True)
    avg_ms = benchmark_inference(model, inf_dl, device)

    # GFLOPs — approximate from training logs or estimate
    gflops = "N/A"

    print('\n' + '=' * 60, flush=True)
    print('PanNuke Fold3 Evaluation Results (LSP-DETR Protocol)')
    print('=' * 60)
    print(f'{"Metric":<25} {"Value":>12}')
    print('-' * 37)
    print(f'{"Dice":<25} {metrics["dice"]:>12.4f}')
    print(f'{"Tissue Acc":<25} {metrics["tissue_acc"]:>12.4f}')
    print(f'{"AJI":<25} {metrics["aji"]:>12.4f}')
    print(f'{"AP@0.5":<25} {ap50:>12.4f}')
    print(f'{"AP@0.7":<25} {ap70:>12.4f}')
    print(f'{"AP@0.9":<25} {ap90:>12.4f}')
    print(f'{"AP@0.5:0.05:0.95":<25} {ap50_95:>12.4f}')
    print(f'{"bPQ":<25} {metrics["bpq"]:>12.4f}')
    print(f'{"bMPQ":<25} {metrics["bmpq"]:>12.4f}')
    print(f'{"mPQ":<25} {metrics["mpq"]:>12.4f}')
    print(f'{"mMPQ":<25} {metrics["mmpq"]:>12.4f}')
    print(f'{"F1 (centroid, r=12)":<25} {f12["f1"]:>12.4f}')
    print(f'{"Precision (centroid)":<25} {f12["precision"]:>12.4f}')
    print(f'{"Recall (centroid)":<25} {f12["recall"]:>12.4f}')
    print(f'{"Params (M)":<25} {n_params / 1e6:>12.2f}')
    print(f'{"GFLOPs":<25} {str(gflops):>12}')
    print(f'{"Inference Time (ms/img)":<25} {avg_ms:>12.2f}')
    print('=' * 37)

    print(f'\nImages evaluated: {len(diag_results)}')

    # Tissue-origin breakdown
    try:
        _print_tissue_breakdown(diag_results, metrics)
        _print_nuclei_breakdown(diag_results, args.num_classes)
    except Exception as exc:
        print(f'\n[DIAG ERROR] {exc}', flush=True)

    # Recall diagnosis
    try:
        _print_recall_diagnosis(diag_results, num_classes=args.num_classes)
    except Exception as exc:
        import traceback
        print(f'\n[RECALL DIAG ERROR] {exc}', flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
