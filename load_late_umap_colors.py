import napari
import numpy as np

FRAME_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_420.csv'
FRAME_UMAP = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_420_umap.csv'
FRAME_COLOR = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_late/140521_late_te_all_tracks_420_color.npy'

face_color = np.load(FRAME_COLOR)
viewer = napari.Viewer()

viewer.open(FRAME_NODES, layer_type='points', size=20, face_color=face_color)
viewer.open(FRAME_UMAP, layer_type='points', size=1, face_color=face_color, visible=False)

napari.run()
