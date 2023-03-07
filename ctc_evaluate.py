import os
from posixpath import split
from re import L
import subprocess
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
import typer
from visualize_lp_solution import load_tiff_frames
from ctc_timings import extract_im_centers
from flow_graph import FlowGraph
from ctc_edge_accuracy import load_lp_solution
from skimage import io

MEASURE_ROOT = "/home/draga/PhD/software/ctc_evaluation/Linux"
DET_SCORE = os.path.join(MEASURE_ROOT, "DETMeasure")
SEG_SCORE = os.path.join(MEASURE_ROOT, "SEGMeasure")
TRA_SCORE = os.path.join(MEASURE_ROOT, "TRAMeasure")

DATA_DIR = "/home/draga/PhD/software/ctc_evaluation/testing_dataset"
SEQ = "01"
NUM_DIGITS = "3"


app = typer.Typer()

def load_solution_graph(seg_ims, solution_path, preserve_inconsistencies=False):
    # load solution so that we have
    # coords with [index, t, (z), y, x]
    # list of edges [(index_src, index_dest)]
    coords, min_t, max_t, corners = extract_im_centers(seg_ims)
    graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)
    solution_edges = load_lp_solution(solution_path, graph, coords, preserve_inconsistencies)

    # we keep the graph structure here so we can dfs search it easily
    # TODO: should just delete edges with no flow set
    graph._g.delete_edges()
    graph._g.add_edges(solution_edges)
    return coords, graph

def assign_track_ids(coords, graph, preserve_inconsistencies=False):
    roots = graph._g.vs.select(
        lambda v: len(graph._g.incident(v, "in")) == 0
        and len(graph._g.incident(v, "out")) > 0
    )
    track_id = 1
    parent_id = 0
    coords["track_id"] = [-1 for _ in range(len(coords))]
    coords["parent_id"] = [0 for _ in range(len(coords))]
    problem = {}
    split_info = ""
    merge_info = ""
    seen_merge = set()
    for root in tqdm(roots, total=len(roots), desc="Traversing graph"):
        for v, _, parent in graph._g.dfsiter(root, advanced=True):
            if coords.loc[[v.index], ["track_id"]].values[0] != -1:
                # print(f'Skipping already visited: {v.index}')
                continue
            # this vertex has a parent so we need to get its parent connection
            if (nparents := v.degree("in")) > 1:
                if v.index not in seen_merge:
                    merge_info += f'Vertex {v.index} has {nparents} incoming edges: {graph._g.incident(v, "in")}\n'
                    seen_merge.add(v.index)
            if parent:
                parent_id = coords.loc[[parent.index], ["track_id"]].values[0]
                n_children = parent.degree("out")
                # this isn't just part of a track, it's a division, so we need a new track id
                if n_children >= 2:
                    track_id += 1
                # if there's too many children and we've been told to fix this
                if n_children > 2 and not preserve_inconsistencies:
                    pid = parent.index
                    problem[pid] = problem.get(pid, 0) + 1
                    # this is the "third child" so we can't parent it to its "real" parent. Make it a new track
                    if problem[pid] >= 3:
                        split_info += f'Parent {pid} has {n_children} children. New track id: {track_id + 1}\n'
                        parent_id = 0
            coords.loc[[v.index], ["track_id"]] = track_id
            coords.loc[[v.index], ["parent_id"]] = parent_id
        track_id += 1
        parent_id = 0
    print(split_info)
    print(merge_info)
            
def mask_ims(coords, graph, seg_ims):
    track_ids = []
    min_frames = []
    max_frames = []
    parents = []
    track_ims = np.zeros_like(seg_ims)

    # for each track_id
    for i in tqdm(range(1, coords.track_id.max()+1), desc="Masking ims"):
        relevant_rows = coords[coords.track_id == i]
        min_frame = relevant_rows["t"].min()
        max_frame = relevant_rows["t"].max()
        parent = relevant_rows["parent_id"].min()

        track_ids.append(i)
        min_frames.append(min_frame)
        max_frames.append(max_frame)
        parents.append(parent)

        for _, row in relevant_rows.iterrows():
            row_coords = tuple(row[graph.spatial_cols].to_numpy(dtype=int))
            t = int(row["t"])
            current_slice = seg_ims[t]
            orig_label = current_slice[row_coords]
            if orig_label == 0:
                orig_label = current_slice[nearest_nonzero_idx(current_slice, row_coords)]
            mask = current_slice == orig_label
            track_ims[t][mask] = row["track_id"]

    # save res_track.txt
    res_track_df = pd.DataFrame(
        {
            "track_id": track_ids,
            "start_frame": min_frames,
            "end_frame": max_frames,
            "parent": parents,
        }
    )
    res_track_df = res_track_df.astype( dtype={'track_id' : int, 
                 'start_frame': int,
                 'end_frame': int,
                 'parent': int})
    return res_track_df, track_ims

def nearest_nonzero_idx(a, idx):
    idx = np.asarray(idx)
    non_zero_indices = np.asarray(np.nonzero(a))
    min_idx = ((non_zero_indices - np.broadcast_to(idx[:, np.newaxis], non_zero_indices.shape))**2).sum(axis=0).argmin()
    return tuple(non_zero_indices[:, min_idx])

def save_masks(masks, root_path):
    for t in tqdm(range(len(masks)), total=len(masks), desc="Saving TIFs"):
        filename = f"mask{t:03d}.tif"
        filepath = os.path.join(root_path, filename)
        im = masks[t]
        tifffile.imwrite(filepath, im, compression=("zlib", 1))

def get_inter_track_edges(res_track, coords):
    src_node_indices = []
    target_node_indices = []

    src_ts = []
    target_ts = []

    src_track_ids = []
    target_track_ids = []

    tracks_with_parents = res_track[res_track.parent != 0]

    for _, track in tracks_with_parents.iterrows():
        parent_track = res_track[res_track.track_id == track['parent']]
        parent_track_id = parent_track['track_id'].values[0]
        parent_end_frame = parent_track['end_frame'].values[0]

        child_track_id = track['track_id']
        child_start_frame = track['start_frame']

        src_node_id = coords[(coords.track_id == parent_track_id) & (coords.t == parent_end_frame)].index[0]
        target_node_id = coords[(coords.track_id == child_track_id) & (coords.t == child_start_frame)].index[0]
        
        src_node_indices.append(src_node_id)
        target_node_indices.append(target_node_id)
        
        src_ts.append(parent_end_frame)
        target_ts.append(child_start_frame)

        src_track_ids.append(parent_track_id)
        target_track_ids.append(child_track_id)
    
    it_edges = pd.DataFrame({
        'src_node_id' : src_node_indices,
        'target_node_id': target_node_indices,
        'src_t': src_ts,
        'target_t': target_ts,
        'src_track_id': src_track_ids,
        'target_track_id': target_track_ids,
    })
    return it_edges

@app.command()
def seg(data_dir, seq):
    # would love not to have to sudo here...
    cmd = ["sudo", SEG_SCORE, data_dir, seq, NUM_DIGITS]
    subprocess.run(cmd)


@app.command()
def det(data_dir, seq):
    # would love not to have to sudo here...
    cmd = ["sudo", DET_SCORE, data_dir, seq, NUM_DIGITS]
    subprocess.run(cmd)


@app.command()
def tra(data_dir, seq):
    # would love not to have to sudo here...
    cmd = ["sudo", TRA_SCORE, data_dir, seq, NUM_DIGITS]
    subprocess.run(cmd)


@app.command()
def convert_gt(data_dir, seq, solution_path):
    # make RES folder if it doesn't exist
    res_path = os.path.join(data_dir, f"{seq}_RES/")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # load TRA GT into array (could actually be TRA also)
    seg_gt_path = os.path.join(data_dir, f"{seq}_GT", "TRA/")
    seg_ims = load_tiff_frames(seg_gt_path)

    # load solution
    coords, graph = load_solution_graph(seg_ims, solution_path)

    # traverse graph and assign track IDs along the way
    assign_track_ids(coords, graph)

    # TODO: what do -1s mean here
    res_track_df, track_ims = mask_ims(coords, graph, seg_ims)
    res_track_path = os.path.join(res_path, "res_track.txt")
    res_track_df.to_csv(res_track_path, header=None, sep=" ", index=False)

    # save mask tifs
    save_masks(track_ims, res_path)

@app.command()
def convert_st(data_dir, seq, solution_path, preserve_inconsistencies=False):
    res_path = os.path.join(data_dir, f"{seq}_RES" + ("_IC/" if preserve_inconsistencies else "/"))
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    seg_st_path = os.path.join(data_dir, f"{seq}_ST", "SEG/")
    seg_ims = load_tiff_frames(seg_st_path)
    coords, graph = load_solution_graph(seg_ims, solution_path, preserve_inconsistencies)
    assign_track_ids(coords, graph, preserve_inconsistencies)

    res_track_df, track_ims = mask_ims(coords, graph, seg_ims)
    res_track_path = os.path.join(res_path, "res_track.txt")
    res_track_df.to_csv(res_track_path, header=None, sep=" ", index=False)

    save_masks(track_ims, res_path)

    it_edges = get_inter_track_edges(res_track_df, coords)
    it_edges_path = os.path.join(res_path, 'it_edges.csv')
    it_edges.to_csv(it_edges_path, index=False)

@app.command()
def convert_stardist(seg_path, data_dir, seq, solution_path):
    # make RES folder if it doesn't exist
    res_path = os.path.join(data_dir, f"{seq}_RES/")
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    seg_ims = io.imread(seg_path).astype(np.uint16)
    coords, graph = load_solution_graph(seg_ims, solution_path)
    assign_track_ids(coords, graph)
    res_track_df, track_ims = mask_ims(coords, graph, seg_ims)
    res_track_path = os.path.join(res_path, "res_track.txt")
    res_track_df.to_csv(res_track_path, header=None, sep=" ", index=False)

    # save mask tifs
    save_masks(track_ims, res_path)

if __name__ == "__main__":
    # app()

    # convert_gt(
    #     "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/",
    #     "02",
    #     "/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/02_GT/output/28Feb23_0731.sol",
    # )

    # convert_stardist(
    #     '/home/draga/PhD/code/repos/misc-scripts/ctc/Fluo-N2DL-HeLa/01/labels.tif',
    #     '/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/',
    #     '01',
    #     '/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/01/output/07Feb23_0831.sol'
    # )

    # convert_st(
    #     "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/",
    #     "02",
    #     "/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/02_ST/output/28Feb23_0730.sol",
    # )

    convert_st(
        "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/",
        "01",
        "/home/draga/PhD/code/experiments/ctc/Fluo-N2DL-HeLa/01_ST/output/07Feb23_0843.sol",
        True
    )

    # a = np.asarray([[3, 2, 3, 3, 0, 2, 4, 2, 1],
    #    [0, 3, 4, 3, 4, 3, 3, 2, 0],
    #    [1, 3, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 1, 2, 0, 0, 0, 0, 0, 2],
    #    [3, 0, 0, 0, 0, 0, 0, 0, 1],
    #    [0, 0, 2, 2, 4, 4, 3, 4, 3],
    #    [2, 2, 2, 1, 0, 0, 1, 1, 1],
    #    [3, 4, 3, 1, 0, 4, 0, 4, 2]])
    # idx = (3, 5)

    # a = np.zeros(shape=(10, 10, 10))
    # a[3:5, 3:5, 3:5] = 1
    # a[8:, 9:, 9: ] = 2
    # idx = (7, 6, 6)
    # nearest = nearest_nonzero_idx(a, idx)
    # print(nearest)
