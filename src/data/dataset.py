from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, resize, normalize


class AtmaDataset(Dataset):
    def _load_image(self, path: Path) -> np.uint8:
        pil_image = Image.open(path).convert("RGB")
        return np.array(pil_image, dtype=np.uint8)

    def __init__(
        self,
        data_csv: Path,
        image_dir: Path,
        transform=None,
        categories: list[str] = [],
    ) -> None:
        self.data_df = pd.read_csv(data_csv)
        self.image_dir = image_dir
        self.transform = transform
        self.categories = ["sorting_date"] + categories

    def __len__(self) -> int:
        return self.data_df.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        object_id = self.data_df.loc[idx, "object_id"]
        label = self.data_df.loc[idx, self.categories].values.astype(np.float32)
        image_path = self.image_dir / f"{object_id}.jpg"
        image = self._load_image(image_path)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = to_tensor(image)
            image = resize(image, size=(224, 224))
            image = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label


class AtmaTestDataset(AtmaDataset):
    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        object_id = self.data_df.loc[idx, "object_id"]
        image_path = self.image_dir / f"{object_id}.jpg"
        image = self._load_image(image_path)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = to_tensor(image)
            image = resize(image, size=(224, 224))
            image = normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, object_id
