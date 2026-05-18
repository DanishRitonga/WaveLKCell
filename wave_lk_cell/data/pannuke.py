from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from albumentations.core.composition import TransformsSeqType
from datasets import DatasetDict, concatenate_datasets, load_dataset
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from wave_lk_cell.data.datasets import TestingDataset, TrainingDataset
from wave_lk_cell.data.samplers.weighted_class_and_tissue import (
    WeightedClassAndTissueSampler,
)
from wave_lk_cell.data.utils import collate_fn, format_transform


class PanNuke(LightningDataModule):
    name = "pannuke"

    def __init__(
        self,
        batch_size: int,
        train_fold: list[int] | int | None = None,
        val_fold: int | None = None,
        test_fold: int | None = None,
        num_workers: int = 0,
        num_classes: int = 5,
        train_transforms: TransformsSeqType | None = None,
        eval_transforms: TransformsSeqType | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

    def setup(self, stage: str) -> None:
        ds: DatasetDict = load_dataset("RationAI/PanNuke")
        ds.set_transform(
            format_transform,
            columns=["image", "instances", "categories"],
            output_all_columns=True,
        )

        match stage:
            case "fit":
                data = (
                    concatenate_datasets([ds[f"fold{f}"] for f in self.train_fold])
                    if isinstance(self.train_fold, Iterable)
                    else ds[f"fold{self.train_fold}"]
                )
                self.train_dataset = TrainingDataset(
                    data, self.train_transforms, self.num_classes
                )
                self.val_dataset = TrainingDataset(
                    ds[f"fold{self.val_fold}"], self.eval_transforms, self.num_classes
                )
            case "validate":
                self.val_dataset = TrainingDataset(
                    ds[f"fold{self.val_fold}"], self.eval_transforms, self.num_classes
                )
            case "test":
                self.test_dataset = TestingDataset(
                    ds[f"fold{self.test_fold}"], self.eval_transforms, self.num_classes
                )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
            pin_memory=True,
            sampler=WeightedClassAndTissueSampler(
                tissues=np.array(self.train_dataset.data["tissue"]),
                classes=self.train_dataset.data["categories"],
                num_classes=len(
                    self.train_dataset.data.features["categories"].feature.names
                ),
                num_samples=len(self.train_dataset),
            ),
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, list[dict[str, Any]]]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
