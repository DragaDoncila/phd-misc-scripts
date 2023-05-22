from ctc_timings import get_im_centers, get_graph
from ctc_fluo_metrics import filter_to_migration_sol, introduce_gt_labels, assign_intertrack_edges
from visualize_lp_solution import load_tiff_frames
from tqdm import tqdm
from traccuracy import TrackingData, TrackingGraph
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatched
from traccuracy.metrics import CTCMetrics, DivisionMetrics, AOGMMetrics

import json
import networkx as nx
import os
import pandas as pd
import time
import pprint

ROOT_DATA_DIR = '/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/'
ds_summary_df = pd.read_csv('/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/ds_summary.csv', index_col=0)

def load_sol(sol_path, seg_path, gt_ims=None):
    seg = load_tiff_frames(seg_path)
    sol = nx.read_graphml(sol_path, node_type=int)
    filter_to_migration_sol(sol)
    if gt_ims is not None:
        introduce_gt_labels(sol, seg, gt_ims)
    assign_intertrack_edges(sol)
    track_graph = TrackingGraph(sol, label_key='pixel_value')
    track_data = TrackingData(track_graph, seg)
    return track_data

if __name__ == '__main__':
    metrics_pth = '/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/ctc_metrics_final.json'
    
    result_dict = {}
    for i, row in enumerate(ds_summary_df.itertuples(), 1):
        ds_name = row.ds_name
        seq = row.seq
        
        sol_dir = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_RES/'.format(seq))
        sol_pth = os.path.join(sol_dir, 'final_solution.graphml')
        seg_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_ST/SEG/'.format(seq))
        gt_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_GT/TRA/'.format(seq))
        
        key = f'{ds_name}_{seq}'
        if key in result_dict and isinstance(result_dict[key], dict):
            print(f"Skipping {key}. Already computed metrics.")
            continue
        if not os.path.exists(sol_pth):
            print(f"No oracle solution graph available for {key}")
            continue
        else:
            print(f"Computing {ds_name} sequence {seq}")
            try:
                gt_data = load_ctc_data(gt_pth)
                sol_data = load_sol(sol_pth, seg_pth, gt_data.segmentation)

                match = CTCMatched(gt_data, sol_data)
                raw_ctc = CTCMetrics(match)
                res = raw_ctc.results
                sol_graph = sol_data.tracking_graph.graph
                tp_node_count = len([node for node in sol_graph.nodes if sol_graph.nodes[node]['is_tp']])
                tp_edge_count = len([e for e in sol_graph.edges if sol_graph.edges[e]['is_tp']])
                res['tp_nodes'] = tp_node_count
                res['tp_edges'] = tp_edge_count
                result_dict[key] = res
                print(key)
                print(res)
            except Exception as e:
                print(f"Failed to compute metrics for {key}")
                result_dict[key] = str(e)
        with open(metrics_pth, 'w') as f:
            json.dump(result_dict, f)

