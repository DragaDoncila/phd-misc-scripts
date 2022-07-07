"""
This file takes the original track files with tracks specified in 3D+t dimensions and performs umap dimension
reduction into 2D+t while keeping the original track_ids in place
"""
import dask.dataframe as df
import dask.array as da
from tqdm import tqdm
from umap import UMAP

# load original track file
ORIGINAL_TRACKS_PATH = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks.txt'
tracks_df = df.read_csv(ORIGINAL_TRACKS_PATH, sep='\t')

tracks_df = df.read_csv(ORIGINAL_TRACKS_PATH, header=0)
tracks_df.columns = ['t', 'z', 'y', 'x', 'node_id', 'parent', 'track_id', 'node_score', 'edge_score']

min_frame = tracks_df['t'].min().compute().astype(int)
max_frame = tracks_df['t'].max().compute().astype(int)

umapper = UMAP(n_components=2)
max_frame_nodes = tracks_df[tracks_df.t == max_frame]
max_frame_coords = max_frame_nodes[['z', 'y', 'x']].to_dask_array(lengths=True)
umapper.fit(max_frame_coords)

df_stacked = None
for frame in tqdm(range(min_frame, max_frame+1)):
    # get the rows frame by frame
    frame_nodes = tracks_df[tracks_df.t == frame]
    frame_coords = frame_nodes[['z', 'y', 'x']].to_dask_array(lengths=True)
    
    # do dims reduction on the x,y,z, coordinates
    frame_umap = da.from_array(umapper.transform(frame_coords))
    # assign *back* into new columns of dataframe
    frame_nodes = frame_nodes.assign(umap_y=frame_umap[:, 0], umap_x=frame_umap[:, 1])
    # concatenate the rows of the dataframes back together
    if df_stacked is None:
        df_stacked = frame_nodes
    else:
        df_stacked = df.concat([df_stacked, frame_nodes])

# write df out to csv
df_stacked.to_csv('/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_with_umap.csv', single_file=True)