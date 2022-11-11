from functools import partial
from string import whitespace
import napari
import numpy as np
from flow_graph import FlowGraph
import os
import pandas as pd
from napari_graph import DirectedGraph
from napari.layers import Graph
from tifffile import TiffFile
import glob

DS_NAME = "Fluo-N2DL-HeLa/01/"
CENTERS_PATH = os.path.join(
    "/home/draga/PhD/code/repos/misc-scripts/ctc/", DS_NAME, "centers.csv"
)
OUTPUT_PATH = os.path.join("/home/draga/PhD/code/experiments/ctc/", DS_NAME, "output/")
SOLUTION_PATH = os.path.join(OUTPUT_PATH, "24Oct22_1647.sol")

# /home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/01/output/24Oct22_1647.csv
EDGE_CSV_PATH = os.path.join(OUTPUT_PATH, "24Oct22_1647.csv")
DATA_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME)

def peek(im_file):
    with TiffFile(im_file) as im:
        im_shape = im.pages[0].shape
        im_dtype = im.pages[0].dtype
    return im_shape, im_dtype

def load_tiff_frames(im_dir):
    all_tiffs = list(sorted(glob.glob(f'{im_dir}*.tif')))
    n_frames = len(all_tiffs)
    frame_shape, im_dtype = peek(all_tiffs[0])
    im_array = np.zeros((n_frames, *frame_shape), dtype=im_dtype)
    for i, tiff_pth in enumerate(all_tiffs):
        with TiffFile(tiff_pth) as im:
            im_array[i] = im.pages[0].asarray()
    return im_array

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


if __name__ == '__main__':
    # make FlowGraph from centers
    node_df = pd.read_csv(CENTERS_PATH)
    coords = node_df[["t", "y", "x"]]
    min_t = 0
    max_t = coords["t"].max()
    corners = [(0, 0), (1024, 1024)]
    graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)

    # select just the edges (and their adjacent nodes) from the solution
    # read edges with a 1 into a set
    edge_flows = dict()
    migration_edges = set()
    
    with open(SOLUTION_PATH) as f:
        for line in f:
            if line.startswith("e"):
                (var_name, flow) = line.strip().split()
                flow = float(flow)
                if flow > 0:
                    edge_flows[var_name] = flow
                    if graph.MIGRATION_EDGE_REGEX.match(var_name):
                        migration_edges.add(var_name)
    

    # set the flows of all the edges
    used_edge_names = set(edge_flows.keys())
    graph._g.es.set_attribute_values('flow', [edge_flows.get(var_name, 0) for var_name in graph._g.es['var_name']])
    # edge_names = list(graph._g.es['var_name'])
    # flow = list(graph._g.es['flow'])
    # parent_indices = [e.source for e in graph._g.es]
    # parent_coords = graph._g.vs[parent_indices]['coords']
    # child_indices = [e.target for e in graph._g.es]
    # child_coords = list(graph._g.vs[child_indices]['coords'])
    # edge_df = pd.DataFrame({
    #     'edge_name':edge_names,
    #     'flow': flow,
    #     'parent_index': parent_indices,
    #     'parent_coords': parent_coords,
    #     'child_index': child_indices,
    #     'child_coords': child_coords
    # })
    # edge_df.to_csv(EDGE_CSV_PATH)

    # get edge indices in the graph, only use migration edges
    shown_edges = graph._g.es.select(var_name_in=migration_edges)
    edge_indices = np.asarray([(e.source, e.target) for e in shown_edges])
    coords = np.asarray([(v["t"], *v["coords"]) for v in graph._g.vs])

    colour_getter = partial(get_colours_from_node, used_edge_names)
    face_colours, edge_colours = zip(*list(map(colour_getter, graph._g.vs)))
    face_colours = np.asarray(face_colours)
    edge_colours = np.asarray(edge_colours)

    # build napari DirectedGraph
    vis_graph = DirectedGraph(edges=edge_indices, coords=coords)

    # read actual image data
    ims = load_tiff_frames(DATA_PATH)
    # padded_ims = np.stack([np.zeros_like(ims[0]), *ims, np.zeros_like(ims[0])])

    # add & run
    viewer = napari.Viewer()
    viewer.add_image(ims, scale=(5, 1, 1))
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
