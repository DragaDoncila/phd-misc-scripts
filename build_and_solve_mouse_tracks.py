import time
import dask.dataframe as df
from flow_graph import FlowGraph

ALL_TRACKS = '/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'
tracks_data = df.read_csv(ALL_TRACKS)
tracks_coords = tracks_data[['t', 'z', 'y', 'x']]
max_t = tracks_coords['t'].max().compute()
min_t = tracks_coords['t'].min().compute()
print(tracks_coords.min().compute())
print(tracks_coords.max().compute())
im_corners = [(505, 10, 713), (4340, 1964, 2157)]
# tracks_coords_tuples = list(tracks_coords.itertuples(index=False, name=None))
start = time.time()
graph = FlowGraph(im_corners, tracks_coords, min_t=min_t, max_t=max_t)
end = time.time()
build_duration = end - start
print("Build duration: ", build_duration)

print("Writing model...")
graph._to_lp('140521_late_te_all_tracks.lp')
end2 = time.time()
write_duration = end2 - end
print("Write duration: ", write_duration)
