import numpy as np
import dask.array as da
import pandas as pd

pth = "/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_with_umap.csv"

tracks_df = pd.load_csv(pth)
print(tracks_df.head())
