import argparse
from pathlib import Path
import logging
import numpy as np
from data.dataset import AtmaDataset

from net import *

ARCH = {
    "resnet18": {"regressor": ResNet18Regressor},
    "resnet50": {"regressor": ResNet50Regressor},
    "efficientnetb0": {"regressor": EfficientNetB0Regressor},
}


def main(
    image_dir: Path,
    train_csv: Path,
):
    dataset = AtmaDataset(train_csv, image_dir)

    x_tot, x2_tot = [], []
    for i in range(len(dataset)):
        image, label = dataset[i]

        x_tot.append((image / 1.0).reshape(-1, 3).mean(0))
        x2_tot.append(((image / 1.0) ** 2).reshape(-1, 3).mean(0))

    img_avr = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avr ** 2)
    print("--- RGB ---")
    print(f"mean: {img_avr}")
    print(f"std: {img_std}")
    print("--- NORMALIZED ---")
    print(f"mean: {img_avr/255.0}")
    print(f"std: {img_std/255.0}")


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Find dataset mean and std.")
    parser.add_argument(
        "--image_dir", type=Path, default=root_dir / "photos", help="Image directory."
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=root_dir / "train.csv",
        help="Location of train_cvX.csv.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(**vars(args))
