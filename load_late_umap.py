import napari
import dask.dataframe as df

STACKED_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_stacked.csv'
UMAP_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_umap.csv'
TRACKS = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'

viewer = napari.Viewer()
viewer.open(STACKED_NODES, layer_type='points', size=20)

tracks_data = df.read_csv(TRACKS)[['track_id', 't', 'z', 'y', 'x']]
viewer.add_tracks(tracks_data.to_dask_array(lengths=True))
viewer.open(UMAP_NODES, layer_type='points', size=1, visible=False)


napari.run()