from functools import partial
from string import whitespace
import napari
import numpy as np
from flow_graph import FlowGraph
import os
import pandas as pd
from napari_graph import DirectedGraph
from napari.layers import Graph

DS_NAME = "Fluo-N2DL-HeLa/01/"
CENTERS_PATH = os.path.join(
    "/home/draga/PhD/code/repos/misc-scripts/ctc/", DS_NAME, "centers.csv"
)

SOLUTION_PATH = (
    "/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/01/output/24Oct22_1647.sol"
)


def get_colours_from_node(names, v):
    incoming, outgoing = graph._get_incident_edges(v)
    incoming_names = set(incoming["var_name"]).intersection(names)
    outgoing_names = set(outgoing["var_name"]).intersection(names)
    face_colour = (1,1,1,)
    edge_colour = (1,1,1,)

    # incoming appearance flow = face colour green
    if any(list(filter(graph.APPEARANCE_EDGE_REGEX.match, incoming_names))):
        face_colour = (0,0.5,0)

    # incoming migration flow = face colour blue
    if any(list(filter(graph.MIGRATION_EDGE_REGEX.match, incoming_names))):
        face_colour = (0,0,1)

    # incoming division flow = face colour yellow
    if any(list(filter(graph.DIVISION_EDGE_REGEX.match, incoming_names))):
        face_colour = (1,1,0)

    # outgoing migration flow = edge colour blue
    if any(list(filter(graph.MIGRATION_EDGE_REGEX.match, outgoing_names))):
        edge_colour = (0,0,1)

    # outgoing exit flow = edge colour red
    if any(list(filter(graph.EXIT_EDGE_REGEX.match, outgoing_names))):
        edge_colour = (1,0,0)

    return face_colour, edge_colour


# make FlowGraph from centers
node_df = pd.read_csv(CENTERS_PATH)
coords = node_df[["t", "y", "x"]]
min_t = 0
max_t = coords["t"].max()
corners = [(0, 0), (1024, 1024)]
graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)

# select just the edges (and their adjacent nodes) from the solution
# read edges with a 1 into a set
edge_set = set()
with open(SOLUTION_PATH) as f:
    for line in f:
        if line.startswith("e"):
            (var_name, val) = line.strip().split()
            if (
                float(val) >= 1
                # no hypervertices
                and "e_a_" not in var_name
                and "_t" not in var_name
                and "e_s_" not in var_name
                and "e_d" not in var_name
            ):
                edge_set.add(var_name)

# turn sequences into napari graph stuff
# coords of nodes (v*4) or (v*3) array
valid_edges = graph._g.es.select(var_name_in=edge_set)
valid_edge_names = set(valid_edges["var_name"])
edge_indices = np.asarray([(e.source, e.target) for e in valid_edges])
coords = np.asarray([(v["t"], *v["coords"]) for v in graph._g.vs])

colour_getter = partial(get_colours_from_node, valid_edge_names)
face_colours, edge_colours = zip(*list(map(colour_getter, graph._g.vs)))
face_colours = np.asarray(face_colours)
edge_colours = np.asarray(edge_colours)

# build napari DirectedGraph
vis_graph = DirectedGraph(edges=edge_indices, coords=coords)

# add & run
viewer = napari.Viewer()
layer = Graph(
    vis_graph, 
    out_of_slice_display=True,
    ndim=3, 
    scale=(5, 1, 1), 
    size=5, 
    face_color=face_colours,
    edge_color=edge_colours
)
viewer.add_layer(layer)

napari.run()
