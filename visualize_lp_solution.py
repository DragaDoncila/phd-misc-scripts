import napari
import numpy as np
from flow_graph import FlowGraph
import os
import pandas as pd
from napari_graph import DirectedGraph
from napari.layers import Graph

DS_NAME = 'Fluo-N2DL-HeLa/01/'
CENTERS_PATH = os.path.join('/home/draga/PhD/code/repos/misc-scripts/ctc/', DS_NAME, 'centers.csv')

SOLUTION_PATH = '/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/01/output/20Oct22_1328.sol'

# make FlowGraph from centers
node_df = pd.read_csv(CENTERS_PATH)
coords = node_df[['t', 'y', 'x']]
min_t = 0
max_t = coords['t'].max()
corners = [(0, 0), (1024, 1024)]
graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)

# select just the edges (and their adjacent nodes) from the solution
# read edges with a 1 into a set
edge_set = set()
with open(SOLUTION_PATH) as f:
    for line in f:
        if line.startswith('e'):
            (var_name, val) = line.strip().split()
            if int(val) >= 1 and 'e_a_' not in var_name and '_t' not in var_name and 'e_s_' not in var_name:
                edge_set.add(var_name)


# turn sequences into napari graph stuff
# coords of nodes (v*4) or (v*3) array
valid_edges = graph._g.es.select(var_name_in=edge_set)
edge_indices = np.asarray([(e.source, e.target) for e in valid_edges])
coords = np.asarray([(v['t'], *v['coords']) for v in graph._g.vs])

# build napari DirectedGraph
vis_graph = DirectedGraph(edges=edge_indices, coords=coords)
print(vis_graph)

# add & run
viewer = napari.Viewer()
layer = Graph(vis_graph, out_of_slice_display=True, ndim=3, scale=(10, 1, 1))
viewer.add_layer(layer)

napari.run()
