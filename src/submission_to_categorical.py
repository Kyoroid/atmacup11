from pathlib import Path
import numpy as np
import pandas as pd

#%%
infile = Path("../artifacts\submission_v2\submission_v2.csv")
df = pd.read_csv(infile)
df = df.apply(np.round, axis="columns")
df.head()

#%%
df.to_csv("../artifacts/submission_v2/submission_v2_cat.csv", index=False)

#%%
