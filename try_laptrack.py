import time
from laptrack import LapTrack
import pandas as pd
import os
import napari


CENTERS_PATH = '/home/draga/PhD/code/repos/misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'

centers = pd.read_csv(CENTERS_PATH)[["t", "z", "y", "x"]]
print(centers.head())
max_distance = 15
start = time.time()
lt = LapTrack(
    track_dist_metric="sqeuclidean",  # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
    splitting_dist_metric="sqeuclidean",
    merging_dist_metric="sqeuclidean",
    # the square of the cutoff distance for the "sqeuclidean" metric
    track_cost_cutoff=max_distance**2,
    splitting_cost_cutoff=max_distance**2,  # or False for non-splitting case
    merging_cost_cutoff=max_distance**2,  # or False for non-merging case
)
track_df, split_df, merge_df = lt.predict_dataframe(
    centers,
    coordinate_cols=[
        "z",
        "y",
        "x",
    ],  # the column names for the coordinates
    frame_col="t",  # the column name for the frame (default "frame")
    only_coordinate_cols=False,  # if False, returned track_df includes columns not in coordinate_cols.
    # False will be the default in the major release.
    validate_frame=False
)
end = time.time()
print(f"Duration: {end - start} seconds")
v = napari.Viewer()
v.add_points(centers[["t", "z", "y", "x"]])
track_df2 = track_df.reset_index()
v.add_tracks(track_df2[["track_id", "t", "z", "y", "x"]])

napari.run()
