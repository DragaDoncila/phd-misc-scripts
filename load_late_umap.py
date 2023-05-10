import napari
import dask.dataframe as df

STACKED_NODES = '/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_stacked.csv'
TRACKS = '/home/draga/PhD/repos/misc-scripts/140521_late/140521_late_te_all_tracks_with_umap.csv'

viewer = napari.Viewer()

tracks_data = df.read_csv(TRACKS)[['track_id', 't', 'umap_y', 'umap_x']]
tracks_data['umap_y'] = tracks_data['umap_y'].mul(50)
tracks_data['umap_x'] = tracks_data['umap_x'].mul(50)
viewer.add_tracks(tracks_data, scale=(100, 1, 1))

points_data = tracks_data[['t', 'umap_y', 'umap_x']]
viewer.add_points(points_data, size=2, scale=(100, 1, 1))

napari.run()
