"""LKCell-exact PQ metrics — copied from LKCell's cell_segmentation/utils/metrics.py.

Provides `remap_label` and `get_fast_pq` so our metric classes produce numbers
directly comparable to LKCell's reported results.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def remap_label(pred: np.ndarray, by_size: bool = False) -> np.ndarray:
    """Relabel instance map to contiguous integers [1, 2, ..., N].

    Optionally sort by instance size (largest first → label 1).
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)

    if len(pred_id) == 0:
        return pred  # nothing to remap

    if by_size:
        pred_size = []
        for inst_id in pred_id:
            pred_size.append((inst_id, np.sum(pred == inst_id)))
        pred_size.sort(key=lambda x: x[1], reverse=True)
        pred_id = [p[0] for p in pred_size]

    new_pred = np.zeros_like(pred)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_pq(
    true: np.ndarray,
    pred: np.ndarray,
    match_iou: float = 0.5,
) -> tuple[list[float], tuple]:
    """Compute Panoptic Quality — exact LKCell implementation.

    Args:
        true: Ground-truth instance map, shape (H, W). Labels must be contiguous.
        pred: Prediction instance map, shape (H, W). Labels must be contiguous.
        match_iou: IoU threshold for matching GT and prediction instances.

    Returns:
        [dq, sq, pq]: Detection Quality, Segmentation Quality, Panoptic Quality.
        (paired_true, paired_pred, unpaired_true, unpaired_pred): pairing info.
    """
    assert match_iou >= 0.0

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [None]  # index 0 = background placeholder
    for t in true_id_list[1:]:
        true_masks.append(np.array(true == t, np.uint8))

    pred_masks = [None]
    for p in pred_id_list[1:]:
        pred_masks.append(np.array(pred == p, np.uint8))

    # pairwise IoU
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64,
    )

    for true_id in true_id_list[1:]:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1
    else:
        # Munkres (Hungarian) assignment on negative IoU
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
