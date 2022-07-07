import napari
import numpy as np

FRAME_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_gt/140521_gt_extended_tracks_420.csv'
FRAME_TSNE = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_gt/140521_gt_extended_tracks_420_tsne.csv'
FRAME_COLOR = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_gt/140521_gt_extended_tracks_420_color.npy'

face_color = np.load(FRAME_COLOR)
viewer = napari.Viewer()
viewer.open(FRAME_NODES, layer_type='points', size=20, face_color=face_color)
viewer.open(FRAME_TSNE, layer_type='points', size=30, face_color=face_color, visible=False)

napari.run()
