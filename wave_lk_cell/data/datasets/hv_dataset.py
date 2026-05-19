from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.core.composition import TransformsSeqType
from albumentations.pytorch import ToTensorV2
from datasets import Dataset
from scipy.ndimage import center_of_mass
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


def _get_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax + 1, cmin, cmax + 1


def _compute_hv_map(masks: np.ndarray) -> np.ndarray:
    H, W = masks.shape[-2:]
    h_map = np.zeros((H, W), dtype=np.float32)
    v_map = np.zeros((H, W), dtype=np.float32)
    for i in range(masks.shape[0]):
        inst = masks[i]
        if inst.sum() == 0:
            continue
        r0, r1, c0, c1 = _get_bounding_box(inst)
        r0 = max(r0 - 2, 0)
        c0 = max(c0 - 2, 0)
        r1 = min(r1 + 2, H)
        c1 = min(c1 + 2, W)
        patch = inst[r0:r1, c0:c1]
        if patch.shape[0] < 2 or patch.shape[1] < 2:
            continue
        com = center_of_mass(patch)
        com = (int(com[0] + 0.5), int(com[1] + 0.5))
        y_coords = np.arange(patch.shape[0]) - com[0]
        x_coords = np.arange(patch.shape[1]) - com[1]
        yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        yy[patch == 0] = 0
        xx[patch == 0] = 0
        for arr in (xx, yy):
            neg = arr[arr < 0]
            pos = arr[arr > 0]
            if neg.size > 0:
                arr[arr < 0] /= -neg.min()
            if pos.size > 0:
                arr[arr > 0] /= pos.max()
        h_map[r0:r1, c0:c1][patch > 0] = yy[patch > 0]
        v_map[r0:r1, c0:c1][patch > 0] = xx[patch > 0]
    return np.stack([h_map, v_map])


def _compute_targets(
    masks: np.ndarray, labels: np.ndarray, num_classes: int
) -> dict[str, Tensor]:
    n = masks.shape[0]
    H, W = masks.shape[-2:]
    binary = np.zeros((H, W), dtype=np.float32)
    type_map = np.zeros((H, W), dtype=np.int64)
    hv_map = np.zeros((2, H, W), dtype=np.float32)
    if n > 0:
        binary = masks.sum(axis=0).clip(0, 1).astype(np.float32)
        hv_map = _compute_hv_map(masks)
        for i in range(n):
            ys, xs = np.nonzero(masks[i])
            lbl = int(labels[i]) if i < len(labels) else 0
            type_map[ys, xs] = lbl
    return {
        "binary_map": torch.from_numpy(binary),
        "hv_map": torch.from_numpy(hv_map),
        "type_map": torch.from_numpy(type_map),
    }


class TrainingDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self,
        data: Dataset,
        transforms: TransformsSeqType | None = None,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.data = data
        self.transforms = transforms if isinstance(transforms, A.Compose) else A.Compose(transforms or [])
        self._to_tensor = ToTensorV2()
        self.num_classes = num_classes
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

        keep = np.any(masks, axis=(1, 2)) if n_instances > 0 else np.array([], dtype=bool)
        masks = masks[keep]
        labels = labels[keep]

        targets = _compute_targets(masks, labels, self.num_classes)
        targets["labels"] = torch.from_numpy(labels).long()
        targets["masks"] = torch.from_numpy(masks)
        targets["tissue"] = self.data.features["tissue"].names[sample["tissue"]]

        return image, targets


class TestingDataset(torch.utils.data.Dataset[tuple[Tensor, dict[str, Any]]]):
    def __init__(
        self,
        data: Dataset,
        transforms: TransformsSeqType | None = None,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.data = data
        self.transforms = transforms if isinstance(transforms, A.Compose) else A.Compose(transforms or [])
        self._to_tensor = ToTensorV2(transpose_mask=True)
        self.num_classes = num_classes
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
        masks_tensor = transformed["mask"]

        n = masks_tensor.shape[0]
        if isinstance(labels, np.ndarray) and labels.ndim == 1 and labels.shape[0] != n:
            labels = np.zeros(n, dtype=np.uint8)

        targets = _compute_targets(masks_tensor.numpy(), labels, self.num_classes)
        targets["labels"] = torch.from_numpy(labels).long()
        targets["masks"] = masks_tensor
        targets["tissue"] = self.data.features["tissue"].names[sample["tissue"]]

        return transformed["image"], targets
