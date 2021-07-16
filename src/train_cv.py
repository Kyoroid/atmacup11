import argparse
from pathlib import Path
import logging
from train import main


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Train model with cross validation.")
    parser.add_argument(
        "--image_dir", type=Path, default=root_dir / "photos", help="Image directory."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(root_dir / "train_cv{fold}.csv"),
        help="Location of train_cvX.csv.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=str(root_dir / "val_cv{fold}.csv"),
        help="Location of val_cvX.csv.",
    )
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
    for fold in range(args.folds):
        main(
            train_csv=Path(args.train_csv.format(fold=fold)),
            val_csv=Path(args.val_csv.format(fold=fold)),
            image_dir=args.image_dir,
            max_epochs=args.max_epochs,
            logdir=args.logdir,
        )
