from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def _sobel_gradients(hv_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = ndimage.sobel(hv_map, axis=1, mode="constant", cval=0.0)
    gy = ndimage.sobel(hv_map, axis=0, mode="constant", cval=0.0)
    return gx, gy


def _marker_controlled_watershed(
    binary_map: np.ndarray,
    horizontal_gradients: np.ndarray,
    vertical_gradients: np.ndarray,
    hv_map: np.ndarray,
) -> np.ndarray:
    sobel_h = ndimage.sobel(hv_map[..., 0], axis=1, mode="constant", cval=0.0)
    sobel_v = ndimage.sobel(hv_map[..., 1], axis=0, mode="constant", cval=0.0)
    edge_strength = np.sqrt(sobel_h**2 + sobel_v**2)

    fg_mask = binary_map > 0.5
    if not fg_mask.any():
        return np.zeros_like(binary_map, dtype=np.int32)

    distance = ndimage.distance_transform_edt(fg_mask)
    coords = ndimage.maximum_filter(distance, size=3)
    local_max = (distance == coords) & fg_mask

    markers, _ = ndimage.label(local_max)
    markers[~fg_mask] = 0

    if markers.max() == 0:
        return np.zeros_like(binary_map, dtype=np.int32)

    edge_normalized = (edge_strength - edge_strength.min()) / (
        edge_strength.max() - edge_strength.min() + 1e-8
    )

    instance_map = ndimage.watershed(edge_normalized, markers, mask=fg_mask)
    return instance_map


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
        result[mask] = majority_class

    return result


def post_process(
    np_binary_map: np.ndarray,
    hv_map: np.ndarray,
    type_map: np.ndarray,
    num_classes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    binary_mask = np_binary_map > 0.5
    instance_map = _marker_controlled_watershed(binary_mask, None, None, hv_map)
    type_instance_map = _majority_voting(instance_map, type_map, num_classes)
    return instance_map, type_instance_map


def post_process_batch(
    np_binary_maps: np.ndarray,
    hv_maps: np.ndarray,
    type_maps: np.ndarray,
    num_classes: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    results = []
    for i in range(np_binary_maps.shape[0]):
        inst, typ = post_process(
            np_binary_maps[i],
            hv_maps[i].transpose(1, 2, 0),
            type_maps[i].transpose(1, 2, 0).argmax(axis=-1),
            num_classes,
        )
        results.append((inst, typ))
    return results
