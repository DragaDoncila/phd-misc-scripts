import napari

STACKED_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_gt/140521_gt_extended_tracks_stacked.csv'
TSNE_NODES = '/home/doncilapopd@hhmi.org/phd-misc-scripts/140521_gt/140521_gt_extended_tracks_tsne.csv'

viewer = napari.Viewer()
viewer.open(STACKED_NODES, layer_type='points', size=20)
viewer.open(TSNE_NODES, layer_type='points', size=30, visible=False)


napari.run()