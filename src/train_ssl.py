import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from data.datamodule import AtmaDataModule, AtmaSimSiamDataModule
from net import *

ARCH = {
    "resnet18_simsiam": {"regressor": ResNet18SimSiamRegressor},
    "vit_dino": {"regressor": ViTDinoRegressor},
}


def main(
    architecture: str,
    image_dir: Path,
    train_csv: Path,
    val_csv: Path,
    max_epochs: int,
    logdir: Path,
    seed: int,
    batch_size: int,
):
    pl.seed_everything(seed)
    if architecture == "resnet18_simsiam":
        datamodule = AtmaSimSiamDataModule(
            image_dir, train_csv, val_csv, batch_size=batch_size
        )
        model: ResNet18SimSiamRegressor = ARCH[architecture]["regressor"](
            learning_rate=0.05 * batch_size / 256
        )
    elif architecture == "vit_dino":
        datamodule = AtmaDataModule(
            image_dir, train_csv, val_csv, batch_size=batch_size
        )
        model: ViTDinoRegressor = ARCH[architecture]["regressor"](
            learning_rate=0.05 * batch_size / 256
        )

    logger = loggers.TestTubeLogger(logdir, name=architecture)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor=None,
        filename="epoch={epoch}-train_loss={loss/train:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=100,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        limit_val_batches=0,
    )
    trainer.fit(model, datamodule=datamodule)


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18_simsiam",
        help="Base architecture.",
        choices=[
            "resnet18_simsiam",
            "vit_dino",
        ],
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
        "--max_epochs", type=int, default=1000, help="Max number of epochs."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--logdir", type=Path, default="./logs", help="Path to save logs."
    )
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
