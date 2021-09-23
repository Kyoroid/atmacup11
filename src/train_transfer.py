import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from data import AtmaDataModule
from net import *

ARCH = {
    "resnet18": {"regressor": ResNet18TransferRegressor},
}


def main(
    architecture: str,
    image_dir: Path,
    train_csv: Path,
    val_csv: Path,
    max_epochs: int,
    logdir: Path,
    seed: int,
    init_lr: int,
    ckpt_path: Path = None,
):
    pl.seed_everything(seed)
    datamodule = AtmaDataModule(image_dir, train_csv, val_csv, batch_size=64)
    model = ResNet18TransferRegressor(learning_rate=init_lr, ckpt_path=ckpt_path)

    logger = loggers.TestTubeLogger(logdir, name=architecture)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/train",
        filename="epoch={epoch}-train_loss={loss/train:.2f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(model, datamodule=datamodule)


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Train model from pretrained encoder.")
    parser.add_argument("ckpt_path", type=Path, help="Checkpoint file.")
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        help="Base architecture.",
        choices=["resnet18"],
    )
    parser.add_argument(
        "--init_lr", type=float, default=1e-4, help="Learning rate at epoch 0."
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
        "--max_epochs", type=int, default=100, help="Max number of epochs."
    )
    parser.add_argument(
        "--logdir", type=Path, default="./logs", help="Path to save logs."
    )
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
