from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from wave_lk_cell.data.datasets import TestingDataset, TrainingDataset
from wave_lk_cell.data.samplers.weighted_class_and_tissue import WeightedClassAndTissueSampler
from wave_lk_cell.data.utils import collate_fn, format_transform


class PanNukeData:
    def __init__(
        self,
        batch_size: int = 16,
        train_fold: int | list[int] | None = 1,
        val_fold: int | None = 2,
        test_fold: int | None = 3,
        num_workers: int = 4,
        num_classes: int = 5,
        train_transforms: TransformsSeqType | None = None,
        eval_transforms: TransformsSeqType | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.train_transforms = A.Compose(train_transforms if isinstance(train_transforms, list) else [train_transforms] if train_transforms else [])
        self.eval_transforms = A.Compose(eval_transforms if isinstance(eval_transforms, list) else [eval_transforms] if eval_transforms else [])

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def setup(self, stage: str = "fit") -> None:
        print("Loading PanNuke dataset from HuggingFace...")
        ds: DatasetDict = load_dataset("RationAI/PanNuke", trust_remote_code=True)
        print(f"  Loaded folds: {list(ds.keys())}")
        ds.set_transform(
            format_transform,
            columns=["image", "instances", "categories"],
            output_all_columns=True,
        )

        if "fit" in stage or "train" in stage:
            train_data = (
                concatenate_datasets([ds[f"fold{f}"] for f in self.train_fold])
                if isinstance(self.train_fold, Iterable) and not isinstance(self.train_fold, int)
                else ds[f"fold{self.train_fold}"]
            )
            train_dataset = TrainingDataset(train_data, self.train_transforms, self.num_classes)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                persistent_workers=self.num_workers > 0,
                drop_last=True,
                pin_memory=True,
                sampler=WeightedClassAndTissueSampler(
                    tissues=np.array(train_dataset.data["tissue"]),
                    classes=train_dataset.data["categories"],
                    num_classes=len(train_dataset.data.features["categories"].feature.names),
                    num_samples=len(train_dataset),
                ),
            )

        if "fit" in stage or "val" in stage:
            val_dataset = TrainingDataset(ds[f"fold{self.val_fold}"], self.eval_transforms, self.num_classes)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        if "test" in stage:
            test_dataset = TestingDataset(ds[f"fold{self.test_fold}"], self.eval_transforms, self.num_classes)
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
