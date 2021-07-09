from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def load_train(dataset_dir: Path):
    df = pd.read_csv(dataset_dir / "train.csv")
    return df


def load_materials(dataset_dir: Path):
    df = pd.read_csv(dataset_dir / "materials.csv")
    return df


def load_techniques(dataset_dir: Path):
    df = pd.read_csv(dataset_dir / "techniques.csv")
    return df


def load_test(dataset_dir: Path):
    df = pd.read_csv(dataset_dir / "test.csv")
    return df


#%%
DATASET_DIR = Path("../dataset_atmaCup11")
train_df = load_train(DATASET_DIR)

#%%
train_df.head()
# %%
materials_df = load_materials(DATASET_DIR)
materials_df.groupby("name").count()

#%%
techniques_df = load_techniques(DATASET_DIR)
techniques_df.groupby("name").count()
#%%
