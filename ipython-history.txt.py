# coding: utf-8
import numpy as np
import dask.dataframe as df
from sklearn.manifold import TSNE
from umap import UMAP, plot
from tqdm import tqdm
import warnings
import napari
tracks_df = df.read_csv(TRACKS_PATH, header=0)
TRACKS_PATH = '140521_late_te_all_tracks.txt'
tracks_df = df.read_csv(TRACKS_PATH, header=0)
get_ipython().run_line_magic('cd', 'phd-misc-scripts')
tracks_df = df.read_csv(TRACKS_PATH, header=0)
tracks_df.columns = ['t', 'z', 'y', 'x', 'node_id', 'parent', 'track_id', 'node_score', 'edge_score']
coords = tracks_df[['t', 'z', 'y', 'x']].to_dask_array(lengths=True)
viewer = napari.Viewer()
viewer.add_points(coords, size=1, name='original_nodes')
viewer.layers[-1].size=10
viewer.add_points(coords, name='original_nodes')
current_slice = viewer.dims.current_step
current_slice
current_slice = current_slice[0]
print(viewer.layers[-1].face_color)
print(viewer.layers[-1].face_color.shape)
slice_data = viewer.layers[-1].data[21]
slice_data
data = viewer.layers[-1].data
data.shape
frame_data = np.extract(data[0] == 421, data)
frame_data
frame_data = tracks_df[tracks_df.t == 421]
frame_data
frame_data.length
frame_nodes = frame_data['x', 'y', 'z']
frame_nodes = frame_data[['x', 'y', 'z']].to_dask_array()
frame_nodes
frame_nodes = frame_data[['x', 'y', 'z']].to_dask_array(lengths=True)
frame_nodes
viewer.add_points(frame_nodes)
viewer.layers[-1].data
color = viewer.layers[-1].face_color
umapper = UMAP(n_components=2)
embedded = umapper.fit_transform(viewer.layers[-1].data)
viewer.add_points(embedded, face_color=color, size=1)
viewer.layers[-1].data *= 8
git status
