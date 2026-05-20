"""Post-processing for instance segmentation from NP + HV predictions.

Uses scipy Sobel (3×3) + EDT-based markers for robust watershed segmentation.
This approach works well with early-training predictions where HV maps may be noisy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _marker_controlled_watershed(
    binary_map: np.ndarray,
    hv_map: np.ndarray,
) -> np.ndarray:
    """Watershed instance segmentation from binary mask + HV map."""
    sobel_h = ndimage.sobel(hv_map[..., 0], axis=1, mode="constant", cval=0.0)
    sobel_v = ndimage.sobel(hv_map[..., 1], axis=0, mode="constant", cval=0.0)
    edge_strength = np.sqrt(sobel_h**2 + sobel_v**2)

    fg_mask = binary_map > 0.5
    if not fg_mask.any():
        return np.zeros_like(binary_map, dtype=np.int32)

    distance = ndimage.distance_transform_edt(fg_mask)
    coords = peak_local_max(
        distance,
        min_distance=5,
        labels=fg_mask.astype(np.int32),
    )

    markers = np.zeros_like(binary_map, dtype=np.int32)
    for y, x in coords:
        markers[y, x] = 1
    markers, _ = ndimage.label(markers)
    markers[~fg_mask] = 0

    if markers.max() == 0:
        return np.zeros_like(binary_map, dtype=np.int32)

    edge_normalized = (edge_strength - edge_strength.min()) / (
        edge_strength.max() - edge_strength.min() + 1e-8
    )

    instance_map = watershed(edge_normalized, markers, mask=fg_mask)
    return instance_map


def _majority_voting(
    instance_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Assign type to each instance by majority vote, falling back to 2nd-most if bg wins."""
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
            # Fall back to 2nd-most common if background wins
            class_counts[0] = 0
            majority_class = class_counts.argmax()
        result[mask] = majority_class

    return result


def post_process(
    np_binary_map: np.ndarray,
    hv_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run watershed post-processing to get instance + type maps."""
    instance_map = _marker_controlled_watershed(np_binary_map, hv_map)
    type_instance_map = _majority_voting(instance_map, type_map, num_classes)
    return instance_map, type_instance_map


def post_process_batch(
    np_binary_maps: np.ndarray,
    hv_maps: np.ndarray,
    type_maps: np.ndarray,
    num_classes: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Post-process a batch of predictions."""
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
        )
        results.append((inst, type_map))
    return results
