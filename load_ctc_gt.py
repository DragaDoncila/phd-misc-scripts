
import os
import numpy as np

import pandas as pd
from visualize_lp_solution import load_tiff_frames
from scipy.ndimage import center_of_mass

DS_NAME = "Fluo-N2DL-HeLa/01/"
CENTERS_PATH = os.path.join(
    "/home/draga/PhD/code/repos/misc-scripts/ctc/", DS_NAME, "centers.csv"
)
DATA_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME)
GT_TRACKS_PATH = f"{DATA_PATH.rstrip('/')}_GT/TRA/"

tracks_df = pd.read_csv(os.path.join(GT_TRACKS_PATH, 'man_track.txt'), delimiter=' ', header=None)
tracks_df.columns = ['id', 'frame_start', 'frame_end', 'parent']

track_labels = load_tiff_frames(GT_TRACKS_PATH)
track_coords = {}
for frame in track_labels:
    current_centers = center_of_mass(frame, labels=frame, index=np.unique(frame)[1:])
    print(current_centers)
    break
