"""Post-processing for instance segmentation from NP + HV predictions.

Adapted from LKCell/HoVerNet with a hybrid approach:
- When HV predictions are mature (sufficient gradient signal), use Sobel-based
  boundary detection from HV maps (LKCell's approach).
- When HV predictions are immature (near-zero), fall back to EDT-based markers
  from the binary mask shape.

This ensures non-zero PQ throughout training while producing sharp instance
boundaries once the HV head converges.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _remove_small_objects(pred: np.ndarray, min_size: int = 10, connectivity: int = 1) -> np.ndarray:
    out = pred.copy()
    if min_size <= 0:
        return out
    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_small_mask = component_sizes < min_size
    too_small_mask = too_small_mask[ccs]
    out[too_small_mask] = 0
    return out


def _hv_is_smooth(h_dir_raw: np.ndarray, v_dir_raw: np.ndarray, threshold: float = 0.1) -> bool:
    rng = max(h_dir_raw.max() - h_dir_raw.min(), v_dir_raw.max() - v_dir_raw.min())
    if rng < 0.05:
        return False
    h_norm = cv2.normalize(h_dir_raw, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_norm = cv2.normalize(v_dir_raw, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    lap_var = float(cv2.Laplacian(h_norm, cv2.CV_32F).var() + cv2.Laplacian(v_norm, cv2.CV_32F).var())
    return lap_var < threshold


def _proc_np_hv_sobel(
    pred: np.ndarray,
    object_size: int = 10,
    ksize: int = 21,
) -> np.ndarray:
    """LKCell's HoVerNet-style NP+HV -> instance map using Sobel on HV."""
    pred = np.array(pred, dtype=np.float32)
    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = ndimage.label(blb)[0]
    blb = _remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1.0 - cv2.normalize(
        sobelh, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )
    sobelv = 1.0 - cv2.normalize(
        sobelv, None, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F,
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall_thresh = np.array(overall >= 0.4, dtype=np.int32)
    marker = blb - overall_thresh
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = ndimage.label(marker)[0]
    marker = _remove_small_objects(marker, min_size=object_size)

    if marker.max() == 0:
        return np.zeros(pred.shape[:2], dtype=np.int32)

    return watershed(dist, markers=marker, mask=blb.astype(bool))


def _proc_np_hv_edt(
    pred: np.ndarray,
) -> np.ndarray:
    """EDT-based fallback when HV signal is too weak."""
    pred = np.array(pred, dtype=np.float32)
    blb_raw = pred[..., 0]

    fg_mask = blb_raw >= 0.5
    if not fg_mask.any():
        return np.zeros(pred.shape[:2], dtype=np.int32)

    blb = np.array(fg_mask, dtype=np.int32)
    blb = ndimage.label(blb)[0]
    blb = _remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1
    fg_mask = blb > 0

    distance = ndimage.distance_transform_edt(fg_mask)
    coords = peak_local_max(
        distance, min_distance=5, labels=fg_mask.astype(np.int32),
    )

    markers = np.zeros(pred.shape[:2], dtype=np.int32)
    for y, x in coords:
        markers[y, x] = 1
    markers, _ = ndimage.label(markers)
    markers[~fg_mask] = 0

    if markers.max() == 0:
        return np.zeros(pred.shape[:2], dtype=np.int32)

    sobel_h = ndimage.sobel(pred[..., 1], axis=1, mode="constant", cval=0.0)
    sobel_v = ndimage.sobel(pred[..., 2], axis=0, mode="constant", cval=0.0)
    edge = np.sqrt(sobel_h**2 + sobel_v**2)
    edge_norm = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)

    return watershed(edge_norm, markers, mask=fg_mask)


def _proc_np_hv(
    pred: np.ndarray,
    object_size: int = 10,
    ksize: int = 21,
) -> np.ndarray:
    """Hybrid NP+HV -> instance map.

    Tries LKCell's Sobel-based approach first. Falls back to EDT-based
    approach if Sobel markers are empty (HV signal too weak).
    """
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    if _hv_is_smooth(h_dir_raw, v_dir_raw):
        result = _proc_np_hv_sobel(pred, object_size=object_size, ksize=ksize)
        if result.max() > 0:
            return result

    return _proc_np_hv_edt(pred)


def _majority_voting(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    result = np.zeros_like(instance_map, dtype=np.int32)
    for inst_id in instance_ids:
        mask = instance_map == inst_id
        if not mask.any():
            continue
        pixels = type_map[mask]
        class_counts = np.bincount(pixels, minlength=num_classes)
        majority_class = class_counts.argmax()
        if majority_class == 0 and num_classes > 1:
            class_counts[0] = 0
            majority_class = class_counts.argmax()
        result[mask] = majority_class

    return result


def post_process(
    np_binary_map: np.ndarray,
    hv_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int = 5,
    ksize: int = 21,
    object_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.stack([np_binary_map, hv_map[..., 0], hv_map[..., 1]], axis=-1)
    instance_map = _proc_np_hv(pred, object_size=object_size, ksize=ksize)
    type_instance_map = _majority_voting(instance_map, type_map, num_classes)
    return instance_map, type_instance_map


def post_process_batch(
    np_binary_maps: np.ndarray,
    hv_maps: np.ndarray,
    type_maps: np.ndarray,
    num_classes: int = 5,
    ksize: int = 21,
    object_size: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    for i in range(np_binary_maps.shape[0]):
        hv = hv_maps[i]
        if hv.ndim == 3:
            hv = hv.transpose(1, 2, 0)
        typ = type_maps[i]
        if typ.ndim == 3:
            typ = typ.transpose(1, 2, 0).argmax(axis=-1)
        inst, type_map = post_process(
            np_binary_maps[i], hv, typ,
            num_classes=num_classes, ksize=ksize, object_size=object_size,
        )
        results.append((inst, type_map))
    return results
