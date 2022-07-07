import sys
import napari
import numpy as np
import dask.dataframe as df
from sklearn.manifold import TSNE
from umap import UMAP, plot
from tqdm import tqdm
import warnings

TRACKS_PATH = '140521_late/140521_late_te_all_tracks.txt'

tracks_df = df.read_csv(TRACKS_PATH, header=0)
tracks_df.columns = ['t', 'z', 'y', 'x', 'node_id', 'parent', 'track_id', 'node_score', 'edge_score']
point_coords = tracks_df[['t', 'z', 'y', 'x']].to_dask_array(lengths=True)

min_frame = tracks_df['t'].min().compute().astype(int)
max_frame = tracks_df['t'].max().compute().astype(int)

tsne_stack = []
umap_stack = []
tsne = TSNE(init='pca', n_components=2, learning_rate='auto')

umapper = UMAP(n_components=2)
max_frame_nodes = tracks_df[tracks_df.t == max_frame]
max_frame_coords = max_frame_nodes[['z', 'y', 'x']].to_dask_array(lengths=True)
umapper.fit(max_frame_coords)
warnings.filterwarnings('ignore')    

for frame in tqdm(range(min_frame, max_frame+1)):
    frame_nodes = tracks_df[tracks_df.t == frame]
    frame_coords = frame_nodes[['z', 'y', 'x']].to_dask_array(lengths=True)

    # frame_tsne = tsne.fit_transform(frame_coords)
    # frame_tsne_coords = np.insert(frame_tsne, 0, frame, axis=1)
    # tsne_stack.extend(frame_tsne_coords)
    
    frame_umap = umapper.transform(frame_coords)
    frame_umap *= 8
    frame_umap_coords = np.insert(frame_umap, 0, frame, axis=1)
    umap_stack.extend(frame_umap_coords)

viewer = napari.Viewer()
# pts = viewer.add_points(point_coords, ndim=4, scale=(5, 1, 1, 1), size=20)
# tsne_embeddings = viewer.add_points(tsne_stack*2, name='tSNE', ndim=3, size=40)
umap_embeddings = viewer.add_points(umap_stack, name='umap', ndim=3, size=1)

napari.run()
