import networkx as nx
import numpy as np
import pandas as pd
from visualize_lp_solution import load_tiff_frames
import napari
from napari_graph import DirectedGraph
from napari.layers import Graph

def assign_track_id(sol):
    """Assign unique integer track ID to each node. 

    Nodes that have more than one incoming edge, or more than
    two children get assigned track ID -1.

    Args:
        sol (nx.DiGraph): directed solution graph
    """
    roots = [node for node in sol.nodes if sol.in_degree(node) == 0]
    nx.set_node_attributes(sol, -1, 'track-id')
    track_id = 1
    for root in roots:
        for edge_key in nx.dfs_edges(sol, root):
            source, dest = edge_key[0], edge_key[1]
            source_out = sol.out_degree(source)
            # true root
            if sol.in_degree(source) == 0:
                sol.nodes[source]['track-id'] = track_id
            # merge into dest or triple split from source
            elif sol.in_degree(dest) > 1 or source_out > 2:
                sol.nodes[source]['track-id'] = -1
                sol.nodes[dest]['track-id'] = -1
                continue
            # double parent_split
            elif source_out == 2:
                track_id += 1
            sol.nodes[dest]['track-id'] = track_id
        track_id += 1

def mask_by_id(nodes, seg):
    masks = np.zeros_like(seg)
    max_id = nodes['track-id'].max()
    for i in range(1, max_id+1):
        track_nodes = nodes[nodes['track-id'] == i]
        for row in track_nodes.itertuples():
            t = row.t
            orig_label = row.label
            mask = seg[t] == orig_label
            masks[t][mask] = row._11 + 1
    
    # colour weird vertices with 1
    unassigned = nodes[nodes['track-id'] == -1]
    for row in unassigned.itertuples():
        t = row.t
        orig_label = row.label
        mask = seg[t] == orig_label
        masks[t][mask] = 1

    return masks

def get_point_colour(sol, merges, bad_parents):
    merges = set(merges)
    bad_parents = set(bad_parents)
    colours = ['white' for _ in range(sol.number_of_nodes())]
    for node in merges:
        parents = [edge[0] for edge in sol.in_edges(node)]
        children = [edge[1] for edge in sol.out_edges(node)]

        # colour the parents orange
        for parent in parents:
            colours[parent] = 'orange'
        # colour the merge node red
        colours[node] = 'red'
        # colour the children yellow
        for child in children:
            colours[child] = 'yellow'

    for node in bad_parents:
        children = [edge[1] for edge in sol.out_edges(node)]
        # colour children pink
        for child in children:
            colours[child] =  'pink'
        # colour parent purple
        colours[node] = 'purple'
    return colours


if __name__ == '__main__':
    seg = load_tiff_frames('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_ST/SEG/')
    data = load_tiff_frames('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01/')
    truth = load_tiff_frames('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_GT/TRA/')
    
    sol = nx.read_graphml('/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_RES_IC/full_sol.graphml', node_type=int)
    assign_track_id(sol)
    node_df = pd.DataFrame.from_dict(sol.nodes, orient='index')
    sol_mask = mask_by_id(node_df, seg)

    merges = [node for node in sol.nodes if sol.in_degree(node) > 1]
    bad_parents = [node for node in sol.nodes if sol.out_degree(node) > 2]

    merge_edges = [edge for node in merges for edge in sol.in_edges(node)]
    merge_edges.extend([edge for node in merges for edge in sol.out_edges(node)])
    
    colors = get_point_colour(sol, merges, bad_parents)

    coords_df = node_df[['t', 'y', 'x']]
    merge_graph = DirectedGraph(edges=merge_edges, coords=coords_df)
    layer = Graph(
        merge_graph, 
        out_of_slice_display=True,
        ndim=3, 
        # scale=(50, 1, 1), 
        size=5, 
        properties=node_df,
        face_color=colors,
    )

    full_graph = DirectedGraph(edges=list(sol.edges.keys()), coords=coords_df)
    full_layer = Graph(
        full_graph, 
        out_of_slice_display=True,
        ndim=3, 
        # scale=(50, 1, 1), 
        size=5, 
        properties=node_df,
        face_color=colors,
    )

    viewer = napari.Viewer()
    viewer.add_image(data)
    viewer.add_labels(
        seg,
        # scale=(50, 1, 1), 
    )
    viewer.add_labels(
        sol_mask,
        name='Solution'
    )
    viewer.add_labels(truth)
    viewer.add_layer(layer)
    viewer.add_layer(full_layer)

    napari.run()
