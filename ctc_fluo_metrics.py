import os
import numpy as np
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatched
from traccuracy.metrics import CTCMetrics, DivisionMetrics, AOGMMetrics
from traccuracy import TrackingData, TrackingGraph
from visualize_lp_solution import load_tiff_frames
import networkx as nx
import pprint

DATA_ROOT = '/home/draga/PhD/data/cell_tracking_challenge/'
DS_NAME = 'Fluo-N2DL-HeLa/'
SEQS = ['01_RES', '02_RES']

def assign_intertrack_edges(g):
    """Currently assigns is_intertrack_edge=True for all edges 
    leaving a division vertex

    Args:
        g (nx.DiGraph): directed tracking graph
    """
    nx.set_edge_attributes(g, 0, name='is_intertrack_edge')
    for e in g.edges:
        src, dest = e
        # source has two children
        if len(g.out_edges(src)) > 1:
            g.edges[e]['is_intertrack_edge'] = 1
        # destination has two parents
        if len(g.in_edges(dest)) > 1:
            g.edges[e]['is_intertrack_edge'] = 1

def introduce_gt_labels(g, seg, gt):
    for v in g.nodes:
        v = g.nodes[v]
        current_label = v["pixel_value"]
        if current_label == 0:
            continue
        t = v["t"]
        # gt label
        if not np.any(seg[t] == current_label):
            orig_label = gt[t, int(v["y"]), int(v["x"])]
            mask = gt[t] == orig_label
            seg[t][mask] = current_label

def filter_to_migration_sol(sol):
    unused_es = [e for e in sol.edges if sol.edges[e]['flow'] == 0]
    sol.remove_edges_from(unused_es)
    delete_vs = []
    for v in sol.nodes:
        v_info = sol.nodes[v]
        if v_info['is_appearance'] or\
            v_info['is_target'] or\
                v_info['is_division'] or\
                    v_info['is_source']:
                    delete_vs.append(v)
    sol.remove_nodes_from(delete_vs)

def load_unchanged_solution(seq, gt_ims):
    # original unchanged solution has label but oracle ones have pixel_value...
    pth = os.path.join(DATA_ROOT, DS_NAME, f'{seq}_IC/')
    seg = load_tiff_frames(os.path.join(DATA_ROOT, DS_NAME,  f'{seq[:2]}_ST/SEG/'))
    sol = nx.read_graphml(os.path.join(pth, 'oracle_introduced_near_parent_no_edges.graphml'), node_type=int)
    filter_to_migration_sol(sol)
    introduce_gt_labels(sol, seg, gt_ims)
    assign_intertrack_edges(sol)
    track_graph = TrackingGraph(sol, label_key='pixel_value')
    track_data = TrackingData(track_graph, seg)
    return track_data

def get_normalized_aogm(aogm, match, fn_node_weight=1, fn_edge_weight=1):
    gt_graph = match.gt_data.tracking_graph.graph
    n_nodes = gt_graph.number_of_nodes()
    n_edges = gt_graph.number_of_edges()
    aogm_0 = n_nodes * fn_node_weight + n_edges * fn_edge_weight
    return 1 - min(aogm, aogm_0) / aogm_0

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    for seq in SEQS[:1]:

        gt_data = load_ctc_data(os.path.join(DATA_ROOT, DS_NAME, f'{seq[:2]}_GT', 'TRA/'))
        unchanged = load_unchanged_solution(seq, gt_data.segmentation)
        # fixed = load_ctc_data(os.path.join(DATA_ROOT, DS_NAME, f'{seq}/'))

        unchanged_match = CTCMatched(gt_data, unchanged)
        # fixed_match = CTCMatched(gt_data, fixed)

        unchanged_raw_ctc = CTCMetrics(unchanged_match).results
        # fixed_raw_ctc = CTCMetrics(fixed_match).results

        # unchanged_unweighted_aogm = AOGMMetrics(unchanged_match).results
        # fixed_unweighted_aogm = AOGMMetrics(fixed_match).results
        # unchanged_normalized_aogm = get_normalized_aogm(unchanged_unweighted_aogm['AOGM'], unchanged_match)
        # fixed_normalized_aogm = get_normalized_aogm(fixed_unweighted_aogm['AOGM'], fixed_match)

        # vns, vfp, vfn, efp, efn, ews
        unchanged_asc = AOGMMetrics(unchanged_match, 0, 0, 0, 1, 1.5, 1).results
        # fixed_asc =  AOGMMetrics(fixed_match, 0, 0, 0, 1, 1.5, 1).results
        unchanged_norm_asc = get_normalized_aogm(unchanged_asc['AOGM'], unchanged_match, 0, 1.5)
        # fixed_norm_asc = get_normalized_aogm(fixed_asc['AOGM'], fixed_match, 0, 1.5)
        
        pp.pprint(unchanged_raw_ctc)
        # pp.pprint(unchanged_normalized_aogm)
        pp.pprint(unchanged_norm_asc)

        # pp.pprint(fixed_raw_ctc)
        # pp.pprint(fixed_normalized_aogm)
        # pp.pprint(fixed_norm_asc)
