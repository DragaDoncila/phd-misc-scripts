import napari
import numpy as np
import dask.dataframe as df
import dask.array as da
from sklearn.manifold import TSNE
from umap import UMAP
from tqdm import tqdm
import warnings

GT_PATH = '/home/draga/PhD/repos/linajea_experiments/data/140521/tracks/tracks.txt'
EXTENDED_PATH = '/home/draga/PhD/repos/linajea_experiments/data/140521/tracks/extended_tracks.txt'

tracks_df = df.read_csv(GT_PATH, sep='\t')
tracks_df.columns = ['t', 'z', 'y', 'x', 'node_id', 'parent', 'track_id']
point_coords = tracks_df[['t', 'z', 'y', 'x']].to_dask_array(lengths=True)

min_frame = tracks_df['t'].min().compute().astype(int)
max_frame = tracks_df['t'].max().compute().astype(int)

tsne_stack = []
umap_stack = []
tsne = TSNE(init='pca', n_components=2, learning_rate='auto')
umap = UMAP(n_components=2, min_dist=0.5, spread=5)
warnings.filterwarnings('ignore')    
for frame in tqdm(range(min_frame, min_frame+20)):
    frame_nodes = tracks_df[tracks_df.t == frame]
    frame_coords = frame_nodes[['z', 'y', 'x']].to_dask_array(lengths=True)

    # frame_tsne = tsne.fit_transform(frame_coords)
    # frame_tsne_coords = np.insert(frame_tsne, 0, frame, axis=1)
    # tsne_stack.extend(frame_tsne_coords)
    
    frame_umap = umap.fit_transform(frame_coords)
    frame_umap_coords = np.insert(frame_umap, 0, frame, axis=1)
    umap_stack.extend(frame_umap_coords)


viewer = napari.Viewer()
# pts = viewer.add_points(point_coords, ndim=4, scale=(5, 1, 1, 1), size=20)
# tsne_embeddings = viewer.add_points(tsne_stack, name='tSNE', ndim=3, size=40)
umap_embeddings = viewer.add_points(umap_stack, name='umap', ndim=3, size=1)

napari.run()
