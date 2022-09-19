import time
import dask.dataframe as df
from flow_graph import FlowGraph

ALL_TRACKS = '/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'
tracks_data = df.read_csv(ALL_TRACKS)
tracks_coords = tracks_data[['t', 'z', 'y', 'x']]
max_t = tracks_coords['t'].max().compute()
min_t = tracks_coords['t'].min().compute()
# just randomly added 10 to the biggest values so they wouldn't be "right on the edge".
# This is pretty shocking so we need some other way to bound the nodes
# also they don't start at 0 so it's gonna be so skewed to disappearance...
im_dim = (4340, 1964, 2157)
tracks_coords_tuples = list(tracks_coords.itertuples(index=False, name=None))
start = time.time()
graph = FlowGraph(im_dim, tracks_coords_tuples, min_t=min_t, max_t=max_t)
end = time.time()
build_duration = end - start
print("Build duration: ", build_duration)

