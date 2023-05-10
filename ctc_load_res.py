import os
import pandas as pd
import napari

from ctc_timings import extract_im_centers
from visualize_lp_solution import load_tiff_frames

IM_PATH = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01/"
RES_PATH = "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_RES/"

def load_res_graph(track_masks, res_path=RES_PATH):
    coords, _, _, _ = extract_im_centers(track_masks)
    it_edges = pd.read_csv(os.path.join(res_path, 'it_edges.csv'))
    for _, edge in it_edges.iterrows():
        # get relevant nodes in coords and put in list
        pass

ims = load_tiff_frames(IM_PATH)
track_masks = load_tiff_frames(RES_PATH)
nodes, edges = load_res_graph(track_masks)

viewer = napari.Viewer()
viewer.add_image(ims)
viewer.add_labels(track_masks)

napari.run()
