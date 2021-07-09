import argparse
from pathlib import Path
import logging
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaDataset
from net import ResNet18Regressor


def main(
    train_csv: Path,
    image_dir: Path,
    checkpoint: Path,
    max_epochs: int,
    logdir: Path,
    batch_size: int,
    device: str,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = A.Compose(
        [
            A.Flip(),
            A.RandomRotate90(),
            A.Resize(width=224, height=224),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    dataset = AtmaDataset(train_csv, image_dir, transform=transform)
    train_subset, valid_subset = random_split(dataset, [3937 - 800, 800])
    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=4)
    model = ResNet18Regressor().to(device)

    logger = loggers.TensorBoardLogger(logdir)
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=logger)
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
        default=root_dir / "train.csv",
        help="Location of train.csv.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default="../checkpoint", help="Path to save models."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used for calculation."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
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
