from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaDataset, AtmaTestDataset


class AtmaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_dir: Path,
        train_csv: Path = None,
        val_csv: Path = None,
        test_csv: Path = None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 224, 224)
        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.PadIfNeeded(256, 256),
                A.RandomCrop(224, 224, always_apply=True),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0, p=0.5
                ),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True
                ),
                ToTensorV2(),
            ]
        )
        self.valtest_transform = A.Compose(
            [
                A.PadIfNeeded(256, 256),
                A.CenterCrop(224, 224, always_apply=True),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True
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
