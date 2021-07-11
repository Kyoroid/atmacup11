from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaTestDataset
from net import ResNet18Regressor
from tqdm.auto import tqdm


def main():
    conf = None
    with Path("./config/submission_v1.json").open(mode="r") as f:
        conf = json.load(f)

    device = torch.device(str(conf["device"]) if torch.cuda.is_available() else "cpu")
    valtest_transform = A.Compose(
        [
            A.PadIfNeeded(256, 256),
            A.CenterCrop(224, 224, always_apply=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
            ToTensorV2(),
        ]
    )

    train_csv = Path(conf["train_csv"])
    test_csv = Path(conf["test_csv"])
    submit_csv = Path(conf["submit_csv"])
    n_folds = int(conf["n_folds"])
    image_dir = Path(conf["image_dir"])
    batch_size = int(conf["batch_size"])

    train_gt = pd.read_csv(train_csv, index_col="object_id")
    train_gt["pred_float"] = np.nan
    submit_df = pd.read_csv(test_csv, index_col="object_id")
    for fold in range(n_folds):
        submit_df[f"target_{fold}"] = np.nan
    submit_df["target"] = np.nan

    #%%
    for fold in range(n_folds):
        val_csv = Path(conf[f"fold_{fold}"]["val_csv"])
        ckpt_path = Path(conf[f"fold_{fold}"]["ckpt_path"])

        model = ResNet18Regressor.load_from_checkpoint(checkpoint_path=ckpt_path).to(
            device
        )
        model.eval()

        val_dataset = AtmaTestDataset(val_csv, image_dir, transform=valtest_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=1, shuffle=False
        )
        for image, object_id in tqdm(val_loader, total=len(val_loader)):
            image = image.to(device)
            y_pred = model(image)
            train_gt.loc[object_id, "pred_float"] = (
                y_pred.cpu().detach().numpy().flatten()
            )

        test_dataset = AtmaTestDataset(test_csv, image_dir, transform=valtest_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
        )
        for image, object_id in tqdm(test_loader, total=len(test_loader)):
            image = image.to(device)
            y_pred = model(image)
            submit_df.loc[object_id, f"target_{fold}"] = (
                y_pred.cpu().detach().numpy().flatten()
            )

    fold_columns = [f"target_{fold}" for fold in range(n_folds)]
    submit_df["target_float"] = submit_df[fold_columns].mean(axis=1, skipna=True)

    #%%
    train_gt["pred"] = train_gt["pred_float"].apply(
        lambda x: np.clip((x - 1550) / 100, 0, 3)
    )
    submit_df["target"] = submit_df["target_float"].apply(
        lambda x: np.clip((x - 1550) / 100, 0, 3)
    )

    #%%
    train_score = np.sqrt(
        mean_squared_error(train_gt["target"].values, train_gt["pred"].values)
    )
    print("Predict score from train.csv: {:.04f}".format(train_score))

    submit_df.to_csv(submit_csv, columns=["target"], index=None)


if __name__ == "__main__":
    main()
