import argparse
from pathlib import Path
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.pytorch.transforms import ToTensorV2
from data import AtmaTestDataset
from net import ResNet18Regressor
from tqdm.auto import tqdm


def main(
    test_csv: Path,
    image_dir: Path,
    ckpt_path: Path,
    submit_csv: Path,
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
    test_dataset = AtmaTestDataset(test_csv, image_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    model = ResNet18Regressor.load_from_checkpoint(checkpoint_path=ckpt_path).to(device)
    model.eval()

    submit_df = pd.read_csv(test_csv, index_col="object_id")
    submit_df["target"] = -1

    for image, object_id in tqdm(test_loader, total=len(test_loader)):
        image = image.to(device)
        y_pred = model(image)
        categorical_pred = torch.clamp((y_pred - 1550) / 100, min=0, max=3).to(device)
        label = categorical_pred.cpu().detach().numpy()
        submit_df.loc[object_id, "target"] = label.flatten()

    submit_df.to_csv(submit_csv, columns=["target"], index=None)


def parse_args():
    root_dir = Path("../dataset_atmaCup11")
    parser = argparse.ArgumentParser(description="Make submission file.")
    parser.add_argument(
        "--image_dir", type=Path, default=root_dir / "photos", help="Image directory."
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=root_dir / "test.csv",
        help="Location of test.csv.",
    )
    parser.add_argument(
        "--submit_csv",
        type=Path,
        default="./submit.csv",
        help="Output path of submit.csv.",
    )
    parser.add_argument("ckpt_path", type=Path, help="Checkpoint file.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device used for calculation."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(**vars(args))
