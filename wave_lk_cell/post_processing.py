"""Post-processing matching LKCell's DetectionCellPostProcessor.__proc_np_hv.

Key differences from old implementation:
- Large Sobel kernel (ksize=21 for 40x magnification)
- cv2.normalize on HV channels before Sobel
- Inverted normalized Sobel: 1 - normalize(sobel)
- overall = max(sobelh, sobelv) - (1 - blb)
- GaussianBlur on distance map
- Edge thresholding at 0.4
- Morphological opening with ellipse kernel
- binary_fill_holes before marker generation
- remove_small_objects at multiple stages
- Type assignment with bg fallback
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import measurements, binary_fill_holes
from skimage.segmentation import watershed


def remove_small_objects(pred: np.ndarray, min_size: int = 64) -> np.ndarray:
    out = pred.copy()
    if min_size == 0:
        return out
    if out.dtype == bool:
        ccs = np.zeros_like(pred, dtype=np.int32)
        measurements.label(pred, output=ccs)
    else:
        ccs = out
    component_sizes = np.bincount(ccs.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out


def proc_np_hv(
    pred: np.ndarray,
    object_size: int = 10,
    ksize: int = 21,
) -> np.ndarray:
    """Process NP + HV prediction → instance map. Matches LKCell exactly.

    Args:
        pred: Shape (H, W, 3). Channel 0 = nuclei probability,
              channel 1 = horizontal/x map, channel 2 = vertical/y map.
    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

    sobelh = 1 - cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1 - cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=object_size)

    proced_pred = watershed(dist, markers=marker, mask=blb)
    return proced_pred


def majority_voting_with_fallback(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Majority voting for type assignment — picks 2nd most dominant if bg wins."""
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    result = np.zeros_like(instance_map, dtype=np.int32)
    for inst_id in instance_ids:
        mask = instance_map == inst_id
        if not mask.any():
            continue
        pixels = type_map[mask]
        type_list, type_pixels = np.unique(pixels, return_counts=True)
        type_list = list(zip(type_list.tolist(), type_pixels.tolist()))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0 and len(type_list) > 1:
            inst_type = type_list[1][0]
        result[mask] = inst_type

    return result


def post_process(
    np_binary_map: np.ndarray,
    hv_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int = 5,
    magnification: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    binary_mask = (np_binary_map > 0.5).astype(np.float32)

    object_size = 10 if magnification == 40 else 3
    ksize = 21 if magnification == 40 else 11

    # Stack into (H, W, 3): [nuclei_prob, h_dir, v_dir]
    pred_stack = np.stack([binary_mask, hv_map[..., 0], hv_map[..., 1]], axis=-1)
    instance_map = proc_np_hv(pred_stack, object_size=object_size, ksize=ksize)
    type_instance_map = majority_voting_with_fallback(instance_map, type_map, num_classes)
    return instance_map, type_instance_map


def post_process_batch(
    np_binary_maps: np.ndarray,
    hv_maps: np.ndarray,
    type_maps: np.ndarray,
    num_classes: int = 5,
    magnification: int = 40,
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
            np_binary_maps[i],
            hv,
            typ,
            num_classes,
            magnification,
        )
        results.append((inst, type_map))
    return results
