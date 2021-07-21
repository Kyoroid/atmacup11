import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning import loggers
from torch.utils.data import dataloader
from data import AtmaDataModule
from data.dataset import AtmaDataset
from net import *
from transforms import Denormalize

ARCH = {
    "resnet18": {"regressor": ResNet18Regressor},
    "resnet50": {"regressor": ResNet50Regressor},
    "efficientnetb0": {"regressor": EfficientNetB0Regressor},
}


def main(
    ckpt_path: Path,
    architecture: str,
    image_dir: Path,
    train_csv: Path,
    val_csv: Path,
    logdir: Path,
):
    datamodule = AtmaDataModule(image_dir, train_csv, val_csv, batch_size=64)
    datamodule.setup(stage=None)
    model = ARCH[architecture]["regressor"]()
    model = model.load_from_checkpoint(
        checkpoint_path=ckpt_path, ckpt_path=str(ckpt_path)
    )
    model.eval()
    denom = Denormalize(
        mean=(0.77695272, 0.74355234, 0.67019692),
        std=(0.16900558, 0.16991152, 0.17102272),
    )

    logger = loggers.TestTubeLogger(logdir, name="proj_" + architecture)

    features = list()
    labels = list()
    images = list()
    for val_batch in datamodule.val_dataloader():
        x, y_gt = val_batch
        e = model.encoder(x)
        categorical_gt = torch.round(torch.clamp((y_gt - 1550) / 100, min=0, max=3))
        x_denom = denom(x)
        features.append(e)
        labels.append(categorical_gt)
        images.append(x_denom)
    features = torch.vstack(features)
    labels = torch.vstack(labels)
    images = torch.vstack(images)

    logger.experiment.add_embedding(
        features, labels.to(torch.int32).tolist(), label_img=images
    )


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Project feature maps.")
    parser.add_argument("ckpt_path", type=Path, default=None, help="Checkpoint file.")
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="Base architecture.",
        choices=["resnet18", "resnet50", "efficientnetb0"],
    )
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
        "--logdir", type=Path, default="./logs", help="Path to save logs."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
