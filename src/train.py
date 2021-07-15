import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaDataset
from net import ResNet18Regressor


def main(
    train_csv: Path,
    val_csv: Path,
    image_dir: Path,
    max_epochs: int,
    logdir: Path,
    batch_size: int,
    device: str,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(256, 256),
            A.RandomCrop(224, 224, always_apply=True),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.PadIfNeeded(256, 256),
            A.CenterCrop(224, 224, always_apply=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            ToTensorV2(),
        ]
    )
    # categories = ["ink", "pencil", "watercolor (paint)"]
    train_dataset = AtmaDataset(train_csv, image_dir, transform=train_transform)
    val_dataset = AtmaDataset(val_csv, image_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    model = ResNet18Regressor().to(device)

    logger = loggers.TensorBoardLogger(logdir)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val",
        filename="epoch={epoch}-val_loss={loss/val:.2f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=False,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--image_dir", type=Path, default=root_dir / "photos", help="Image directory."
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=root_dir / "train_cv0.csv",
        help="Location of train_cvX.csv.",
    )
    parser.add_argument(
        "--val_csv",
        type=Path,
        default=root_dir / "val_cv0.csv",
        help="Location of val_cvX.csv.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used for calculation."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Max number of epochs."
    )
    parser.add_argument(
        "--logdir", type=Path, default="./logs", help="Path to save logs."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(**vars(args))
