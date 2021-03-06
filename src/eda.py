from itertools import groupby
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
DATASET_DIR = Path("../../dataset_atmaCup11")
train_df = load_train(DATASET_DIR)

#%%
train_df.head()
# %%
materials_df = load_materials(DATASET_DIR)

#%%
techniques_df = load_techniques(DATASET_DIR)
tq_binary_df = (
    pd.get_dummies(techniques_df, columns=["name"]).groupby("object_id").sum()
)
pd.merge(train_df, tq_binary_df, how="left", on="object_id").fillna(0.5)
#%%
pd.merge(train_df, tq_binary_df, how="left", on="object_id").fillna(0.5).groupby(
    "name_engraving"
).count()
# %%
materials_df = load_materials(DATASET_DIR)
mt_binary_df = pd.get_dummies(materials_df, columns=["name"]).groupby("object_id").sum()
pd.merge(train_df, mt_binary_df, how="left", on="object_id").fillna(0)
# %%
fixed_mt_binary_df = pd.merge(
    train_df, mt_binary_df, how="left", on="object_id"
).fillna(0)
fixed_mt_binary_df.rename(
    columns=lambda x: x[5:] if x[:5] == "name_" else x, inplace=True
)
# %%
fixed_mt_binary_df
#%%
fixed_mt_binary_df.columns
# %%
fixed_mt_binary_df.groupby("target").mean()[
    [
        "India ink (ink)",
        "bristol board",
        "cardboard",
        "chalk",
        "deck paint",
        "gold leaf",
        "gouache (paint)",
        "graphite (mineral)",
        "ink",
        "leather",
        "linen (material)",
        "metal",
        "oil paint (paint)",
    ]
]
#%%
fixed_mt_binary_df.groupby("target").mean()[
    [
        "paint (coating)",
        "palm leaf (material)",
        "paper",
        "parchment (animal material)",
        "pencil",
        "prepared paper",
        "tracing paper",
        "varnish",
        "velvet (fabric weave)",
        "watercolor (paint)",
        "wood (plant material)",
        "zinc",
    ]
]
#%%
