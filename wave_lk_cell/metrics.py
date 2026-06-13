# -*- coding: utf-8 -*-
# Based on: https://github.com/TissueImageAnalytics/PanNuke-metrics

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)  #[256,256]
    pred = np.copy(pred)  #(256,256)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    # if there is no background, fixing by adding it
    if 0 not in pred_id_list:
        pred_id_list = [0] + pred_id_list

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou for all instances
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)  # good practice?
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def masked_pq(true_inst_map, pred_inst_map, iou_threshold=0.5, eps=1e-6):
    """Compute Masked Panoptic Quality (LSP-DETR, https://arxiv.org/abs/2601.03163).

    Standard PQ penalizes predicted pixels overlapping other GT instances.
    Masked PQ masks out those pixels from the prediction, making it fairer
    for overlapping nuclei scenarios.

    For each (GT_i, Pred_j) pair:
      intersection = GT_i & Pred_j
      masked_pred_j = Pred_j & ~any(GT)  (predicted pixels NOT on any GT instance)
      union = sum(GT_i) + sum(masked_pred_j)

    This means predicted pixels that overlap OTHER GT instances are not counted
    as false positives in the union, so they don't penalize IoU.

    Args:
        true_inst_map: (H, W) int array, GT instance map (0=bg)
        pred_inst_map: (H, W) int array, predicted instance map (0=bg)
        iou_threshold: IoU threshold for matching
        eps: small value for numerical stability

    Returns:
        [dq, sq, pq]: measurement statistic
    """
    from scipy.optimize import linear_sum_assignment

    true_id_list = list(np.unique(true_inst_map))
    pred_id_list = list(np.unique(pred_inst_map))
    if 0 in true_id_list:
        true_id_list.remove(0)
    if 0 in pred_id_list:
        pred_id_list.remove(0)
    if len(true_id_list) == 0 and len(pred_id_list) == 0:
        return [1.0, 1.0, 1.0]
    if len(true_id_list) == 0 or len(pred_id_list) == 0:
        return [0.0, 0.0, 0.0]

    t_flat = np.stack([true_inst_map == t for t in true_id_list]).reshape(len(true_id_list), -1).astype(np.float64)
    p_flat = np.stack([pred_inst_map == p for p in pred_id_list]).reshape(len(pred_id_list), -1).astype(np.float64)

    intersection = t_flat @ p_flat.T

    any_true = t_flat.any(axis=0)
    masked_p = p_flat.copy()
    masked_p[:, any_true] = 0.0
    union = t_flat.sum(axis=1, keepdims=True) + masked_p.sum(axis=1, keepdims=True).T
    union = np.maximum(union, eps)

    iou = intersection / union

    row_ind, col_ind = linear_sum_assignment(-iou)
    paired_iou = iou[row_ind, col_ind]
    valid = paired_iou > iou_threshold
    paired_iou = paired_iou[valid]
    tp = len(paired_iou)
    fp = len(pred_id_list) - tp
    fn = len(true_id_list) - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn + eps)
    sq = float(paired_iou.sum()) / (tp + eps) if tp > 0 else 0.0
    pq = dq * sq

    return [dq, sq, pq]


def compute_aji(pred_masks, gt_masks, iou_threshold=0.5):
    """Compute Aggregated Jaccard Index for one image.

    AJI = sum_intersections / (sum_unions + unmatched_pred_area + unmatched_gt_area)

    Uses Hungarian matching at iou_threshold to find optimal pairs.

    Args:
        pred_masks: List of (H, W) uint8 binary masks (predictions).
        gt_masks: List of (H, W) uint8 binary masks (ground truth).
        iou_threshold: Minimum IoU for valid match.

    Returns:
        AJI score in [0, 1]. Returns 0.0 if no GT or pred masks.
    """
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0

    iou_matrix = mask_iou_matrix(pred_masks, gt_masks)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    match_pred = set(row_ind[valid].tolist())
    match_gt = set(col_ind[valid].tolist())

    total_intersection = 0.0
    total_union = 0.0
    for r, c in zip(row_ind[valid], col_ind[valid]):
        p = pred_masks[r].astype(np.float64)
        g = gt_masks[c].astype(np.float64)
        total_intersection += (p * g).sum()
        total_union += (p + g - p * g).sum()

    for i in range(len(pred_masks)):
        if i not in match_pred:
            total_union += pred_masks[i].astype(np.float64).sum()

    for j in range(len(gt_masks)):
        if j not in match_gt:
            total_union += gt_masks[j].astype(np.float64).sum()

    if total_union == 0:
        return 0.0
    return total_intersection / total_union


def mask_iou_matrix(pred_masks, gt_masks):
    """Compute pairwise mask IoU between two lists of binary masks.

    Args:
        pred_masks: List of (H, W) uint8 binary masks.
        gt_masks: List of (H, W) uint8 binary masks.

    Returns:
        IoU matrix of shape (N_pred, N_gt).
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64)

    pred_stack = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float64)
    gt_stack = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float64)

    intersection = pred_stack @ gt_stack.T
    pred_area = pred_stack.sum(axis=1, keepdims=True)
    gt_area = gt_stack.sum(axis=1, keepdims=True)
    union = pred_area + gt_area.T - intersection

    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float64), where=union > 0)


def compute_pq_masked(pred_masks, gt_masks, iou_threshold=0.5, mask=None):
    """Compute PQ with optional foreground mask (bMPQ / mMPQ style).

    Matches RayCastED's _compute_pq_masked exactly.

    Args:
        pred_masks: List of (H, W) uint8 binary masks.
        gt_masks: List of (H, W) uint8 binary masks.
        iou_threshold: IoU threshold for matching.
        mask: Optional foreground mask to apply before matching.

    Returns:
        (pq, sq, dq) tuple.
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    if n_gt == 0 or n_pred == 0:
        return 0.0, 0.0, 0.0

    if mask is not None:
        pred_masks = [m & mask for m in pred_masks]
        gt_masks = [m & mask for m in gt_masks]

    iou_matrix = mask_iou_matrix(pred_masks, gt_masks)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold

    tp = valid.sum()
    fp = n_pred - tp
    fn = n_gt - tp

    dq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
    sq = float(iou_matrix[row_ind[valid], col_ind[valid]].mean()) if tp > 0 else 0.0
    pq = sq * dq
    return float(pq), float(sq), float(dq)


def compute_centroid_f1(pred_centroids, gt_centroids, radius=12.0):
    """Compute centroid-based F1 using Hungarian matching.

    A prediction is a true positive if its centroid is within `radius` pixels
    of a GT centroid after optimal (Hungarian) assignment.

    Args:
        pred_centroids: (N, 2) array of (x, y) prediction centroids.
        gt_centroids: (M, 2) array of (x, y) GT centroids.
        radius: Maximum distance for a valid match (default: 12).

    Returns:
        (tp, fp, fn) tuple.
    """
    n_pred = len(pred_centroids)
    n_gt = len(gt_centroids)

    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    dist = np.linalg.norm(gt_centroids[:, :2][:, None] - pred_centroids[:, :2][None, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist)
    tp = int((dist[row_ind, col_ind] <= radius).sum())
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


def compute_ap(recall, precision):
    """Compute Average Precision from recall and precision arrays.

    Uses all-points interpolation matching COCO/LSP-DETR protocol.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred
