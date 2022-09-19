import time
import dask.dataframe as df
from scipy.spatial import KDTree
from tqdm import tqdm

ALL_TRACKS = '/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'
tracks_data = df.read_csv(ALL_TRACKS)
tracks_coords = tracks_data[['t', 'z', 'y', 'x']]

array_coords = tracks_coords.to_dask_array(lengths=True)
start = time.time()
my_tree = KDTree(array_coords, leafsize=100)
whole_tree_end = time.time()

my_near_points = my_tree.query_ball_point(array_coords[10], 50)
near_query_end = time.time()

per_frame_trees = []
max_t = tracks_coords['t'].max().compute()
min_t = tracks_coords['t'].min().compute()
for t in tqdm(range(min_t, max_t+1)):
    frame_coords = tracks_coords[tracks_coords.t == t]
    frame_coords = frame_coords[['z', 'y', 'x']]
    array_coords = frame_coords.to_dask_array(lengths=True)
    new_tree = KDTree(array_coords)
per_frame_trees.append(new_tree)
per_frame_end = time.time()

print(f'Build duration: {whole_tree_end-start}')
print(f'Query time: {near_query_end-whole_tree_end}')
print(array_coords[10].compute())
print(len(my_near_points))
print(f'Per frame build duration: {per_frame_end - near_query_end}')
