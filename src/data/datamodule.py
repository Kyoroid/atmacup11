from pathlib import Path
from typing import Optional
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import lightly
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaDataset, AtmaTestDataset
from data.collate import BaseCollateFunction


class AtmaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: Path,
        train_csv: Path = None,
        val_csv: Path = None,
        test_csv: Path = None,
        batch_size: int = 64,
        num_workers: int = 4,
        input_size: int = 224,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.dims = (3, input_size, input_size)
        self.train_transform = A.Compose(
            [
                A.Flip(p=0.5),
                A.PadIfNeeded(input_size + input_size, input_size + input_size),
                A.RandomCrop(input_size, input_size, always_apply=True),
                A.Normalize(
                    mean=(0.77695272, 0.74355234, 0.67019692),
                    std=(0.16900558, 0.16991152, 0.17102272),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )
        self.valtest_transform = A.Compose(
            [
                A.PadIfNeeded(input_size + input_size, input_size + input_size),
                A.CenterCrop(input_size, input_size, always_apply=True),
                A.Normalize(
                    mean=(0.77695272, 0.74355234, 0.67019692),
                    std=(0.16900558, 0.16991152, 0.17102272),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = AtmaDataset(
                self.train_csv, self.image_dir, transform=self.train_transform
            )
            self.val_dataset = AtmaDataset(
                self.val_csv, self.image_dir, transform=self.valtest_transform
            )
        elif stage == "test" or stage is None:
            self.test_dataset = AtmaTestDataset(
                self.test_csv, self.image_dir, transform=self.valtest_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class AtmaSimSiamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: Path,
        train_csv: Path = None,
        val_csv: Path = None,
        test_csv: Path = None,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 224,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.dims = (3, input_size, input_size)
        self.train_transform = A.Compose(
            [
                A.Flip(p=0.5),
                A.PadIfNeeded(input_size + input_size, input_size + input_size),
                A.RandomCrop(input_size, input_size, always_apply=True),
                A.Normalize(
                    mean=(0.77695272, 0.74355234, 0.67019692),
                    std=(0.16900558, 0.16991152, 0.17102272),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )
        self.collate_fn = BaseCollateFunction(self.train_transform)
        self.valtest_transform = A.Compose(
            [
                A.PadIfNeeded(input_size + input_size, input_size + input_size),
                A.CenterCrop(input_size, input_size, always_apply=True),
                A.Normalize(
                    mean=(0.77695272, 0.74355234, 0.67019692),
                    std=(0.16900558, 0.16991152, 0.17102272),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def setup(self, stage: Optional[str]) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = lightly.data.LightlyDataset(
                input_dir=self.image_dir,
            )
            self.val_dataset = lightly.data.LightlyDataset(
                input_dir=self.image_dir,
                transform=self.valtest_transform,
            )
        elif stage == "test" or stage is None:
            self.test_dataset = lightly.data.LightlyDataset(
                input_dir=self.image_dir,
                transform=self.valtest_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
