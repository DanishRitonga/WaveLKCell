from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import WeightedRandomSampler


class WeightedClassAndTissueSampler(WeightedRandomSampler):
    def __init__(
        self,
        tissues: NDArray[np.uint8],
        classes: list[NDArray[np.uint8]],
        num_classes: int,
        num_samples: int,
        gamma: float = 0.85,
        replacement: bool = True,
    ) -> None:
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"

        tw = self._get_sampling_weights_tissue(tissues, gamma)
        cw = self._get_sampling_weights_cell(classes, num_classes, gamma)
        weights = tw / tw.max() + cw / cw.max()

        super().__init__(weights.tolist(), num_samples, replacement)

    @staticmethod
    def _get_sampling_weights_tissue(
        tissues: NDArray[np.uint8], gamma: float = 1
    ) -> NDArray[np.float64]:
        _, counts = np.unique(tissues, return_counts=True)
        n = len(tissues)
        weights = n / (gamma * counts + (1 - gamma) * n)
        return weights[tissues]

    @staticmethod
    def _get_sampling_weights_cell(
        classes: list[NDArray[np.uint8]], num_classes: int, gamma: float = 1
    ) -> NDArray[np.float64]:
        binary_cell_counts = np.zeros((len(classes), num_classes), dtype=np.bool)

        for i, arr in enumerate(classes):
            unique_classes = np.unique(arr)
            binary_cell_counts[i, unique_classes] = True

        binary_weight_factors = binary_cell_counts.sum(axis=0)
        n = binary_weight_factors.sum()
        weight_vector = n / (gamma * binary_weight_factors + (1 - gamma) * n)
        img_weight = (1 - gamma) * binary_cell_counts.max(axis=-1) + gamma * np.sum(
            binary_cell_counts * weight_vector, axis=-1
        )
        img_weight[img_weight == 0] = img_weight[img_weight != 0].min()
        return img_weight
