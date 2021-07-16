import argparse
from pathlib import Path
import logging
import pytorch_lightning as pl
from data import AtmaDataModule
from net import *

ARCH = {
    "resnet18": {"regressor": ResNet18Regressor},
    "resnet50": {"regressor": ResNet50Regressor},
    "efficientnetb0": {"regressor": EfficientNetB0Regressor},
}


def main(
    plot_file: Path,
    architecture: str,
    train_csv: Path,
    val_csv: Path,
    image_dir: Path,
    seed: int,
    ckpt_path: Path = None,
):
    pl.seed_everything(seed)
    datamodule = AtmaDataModule(image_dir, train_csv, val_csv)
    model: BaseRegressor = ARCH[architecture]["regressor"]()
    if ckpt_path:
        model = model.load_from_checkpoint(
            checkpoint_path=ckpt_path, ckpt_path=str(ckpt_path)
        )

    trainer = pl.Trainer(
        gpus=1,
    )
    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
    fig = lr_finder.plot(suggest=True)
    suggested_lr = lr_finder.suggestion()
    fig.suptitle(model.__class__.__name__, fontsize=16)
    fig.savefig(plot_file)
    print(f"Best learning rate: {suggested_lr:.3e}")


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Find learning rate.")
    parser.add_argument(
        "plot_file", type=Path, help="File name which will drawn learning rate curve."
    )
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
    parser.add_argument("--seed", type=int, default=2021, help="Random seed.")
    parser.add_argument(
        "--ckpt_path", type=Path, default=None, help="Checkpoint file."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(**vars(args))
