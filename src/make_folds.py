import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def main(
    trainval_csv: Path,
    out_dir: Path,
    logdir: Path,
    folds: int,
):
    trainval_df = pd.read_csv(trainval_csv)
    X = trainval_df.index.values
    groups = trainval_df["art_series_id"].values

    kfold = GroupKFold(n_splits=folds)
    for i, (train_idx, val_idx) in enumerate(kfold.split(X, groups=groups)):
        train_df = trainval_df.iloc[train_idx]
        val_df = trainval_df.iloc[val_idx]
        train_df.to_csv(
            out_dir / f"train_cv{i}.csv", index=None, columns=trainval_df.columns
        )
        val_df.to_csv(
            out_dir / f"val_cv{i}.csv", index=None, columns=trainval_df.columns
        )


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Make folds for cross validation.")
    parser.add_argument(
        "--trainval_csv",
        type=Path,
        default=root_dir / "train.csv",
        help="Location of train.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=root_dir,
        help="Location of output.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds.",
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
