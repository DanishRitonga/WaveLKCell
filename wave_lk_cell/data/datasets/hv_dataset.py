from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset
from torch import Tensor

_GEO_TYPES = (
    A.RandomRotate90, A.HorizontalFlip, A.VerticalFlip,
    A.Transpose, A.ShiftScaleRotate, A.ElasticTransform,
    A.GridDistortion, A.OpticalDistortion, A.RandomCrop,
    A.RandomSizedCrop, A.Crop, A.CropNonEmptyMaskIfExists,
    A.PadIfNeeded, A.Resize, A.LongestMaxSize, A.SmallestMaxSize,
)


def _split_transforms(
    transforms: A.Compose,
) -> tuple[A.Compose | None, A.Compose | None]:
    geo_list = [t for t in transforms.transforms if isinstance(t, _GEO_TYPES)]
    img_list = [t for t in transforms.transforms if not isinstance(t, _GEO_TYPES)]
    return (
        A.Compose(geo_list) if geo_list else None,
        A.Compose(img_list) if img_list else None,
    )


class TrainingDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self,
        data: Dataset,
        transforms: TransformsSeqType | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self._to_tensor = ToTensorV2()
        self.categories = (
            self.data.features["categories"].feature.names
            if "categories" in self.data.features
            else []
        )
        self.geo_transforms, self.img_transforms = _split_transforms(self.transforms)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.data[idx]
        image = sample["image"]
        masks = sample["instances"]
        labels = (
            sample["categories"]
            if "categories" in sample
            else np.zeros(masks.shape[-1])
        )

        if self.geo_transforms is not None:
            transformed = self.geo_transforms(image=image, mask=masks)
            image = transformed["image"]
            masks = transformed["mask"]

        masks = masks.transpose(2, 0, 1)
        n_instances = masks.shape[0]
        if isinstance(labels, np.ndarray) and labels.ndim == 1 and labels.shape[0] != n_instances:
            labels = np.zeros(n_instances, dtype=np.uint8)

        if self.img_transforms is not None:
            image = self.img_transforms(image=image)["image"]

        image = self._to_tensor(image=image)["image"]
        masks = torch.from_numpy(masks)

        keep = masks.any(axis=(1, 2)).bool()
        masks = masks[keep]
        labels = labels[keep.numpy()]

        return image, {
            "masks": masks,
            "labels": torch.from_numpy(labels).long(),
            "tissue": self.data.features["tissue"].names[sample["tissue"]],
        }


class TestingDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self,
        data: Dataset,
        transforms: TransformsSeqType | None = None,
    ) -> None:
        super().__init__()
        self.data = data
        self.transforms = A.Compose(transforms or [])
        self._to_tensor = ToTensorV2(transpose_mask=True)
        self.categories = (
            self.data.features["categories"].feature.names
            if "categories" in self.data.features
            else []
        )
        self.geo_transforms, self.img_transforms = _split_transforms(self.transforms)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        sample = self.data[idx]
        image = sample["image"]
        masks = sample["instances"]
        labels = (
            sample["categories"]
            if "categories" in sample
            else np.zeros(masks.shape[-1])
        )

        if self.geo_transforms is not None:
            transformed = self.geo_transforms(image=image, mask=masks)
            image = transformed["image"]
            masks = transformed["mask"]

        if self.img_transforms is not None:
            image = self.img_transforms(image=image)["image"]

        transformed = self._to_tensor(image=image, mask=masks)

        return transformed["image"], {
            "masks": transformed["mask"],
            "labels": torch.from_numpy(labels).long(),
            "tissue": self.data.features["tissue"].names[sample["tissue"]],
        }
