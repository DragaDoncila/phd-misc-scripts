import os
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

def load_unchanged_solution(seq):
    pth = os.path.join(DATA_ROOT, DS_NAME, f'{seq}_IC/')
    seg = load_tiff_frames(os.path.join(DATA_ROOT, DS_NAME,  f'{seq}_ST/'))
    sol = nx.read_graphml(os.path.join(pth, 'full_sol.graphml'), node_type=int)
    track_graph = TrackingGraph(sol, label_key='label')
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
        unchanged = load_unchanged_solution(seq)
        fixed = load_ctc_data(os.path.join(DATA_ROOT, DS_NAME, f'{seq}/'))

        unchanged_match = CTCMatched(gt_data, unchanged)
        fixed_match = CTCMatched(gt_data, fixed)

        unchanged_raw_ctc = CTCMetrics(unchanged_match).results
        fixed_raw_ctc = CTCMetrics(fixed_match).results

        unchanged_unweighted_aogm = AOGMMetrics(unchanged_match).results
        fixed_unweighted_aogm = AOGMMetrics(fixed_match).results
        unchanged_normalized_aogm = get_normalized_aogm(unchanged_unweighted_aogm['AOGM'], unchanged_match)
        fixed_normalized_aogm = get_normalized_aogm(fixed_unweighted_aogm['AOGM'], fixed_match)

        # vns, vfp, vfn, efp, efn, ews
        unchanged_asc = AOGMMetrics(unchanged_match, 0, 0, 0, 1, 1.5, 1).results
        fixed_asc =  AOGMMetrics(fixed_match, 0, 0, 0, 1, 1.5, 1).results
        unchanged_norm_asc = get_normalized_aogm(unchanged_asc['AOGM'], unchanged_match, 0, 1.5)
        fixed_norm_asc = get_normalized_aogm(fixed_asc['AOGM'], fixed_match, 0, 1.5)
        
        pp.pprint(unchanged_raw_ctc)
        pp.pprint(unchanged_normalized_aogm)
        pp.pprint(unchanged_norm_asc)

        pp.pprint(fixed_raw_ctc)
        pp.pprint(fixed_normalized_aogm)
        pp.pprint(fixed_norm_asc)
