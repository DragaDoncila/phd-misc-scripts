from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatched, IOUMatched
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy import TrackingData, TrackingGraph
from visualize_lp_solution import load_tiff_frames

import networkx as nx
import pprint

gt_data = load_ctc_data(
    "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_GT/TRA/"
)
pred_data = load_ctc_data(
    "/home/draga/PhD/data/cell_tracking_challenge/Fluo-N2DL-HeLa/01_RES/"
)


# delete all inter-track edges
pred_data_no_parent_edges = TrackingData(TrackingGraph(pred_data.tracking_graph.graph.copy()), segmentation=pred_data.segmentation)
tgraph = pred_data_no_parent_edges.tracking_graph
it_edges = tgraph.get_edges_with_attribute('is_intertrack_edge', criterion=lambda x: x)
tgraph.graph.remove_edges_from(it_edges)

# delete all edges
pred_data_no_edges = TrackingData(TrackingGraph(nx.create_empty_copy(pred_data.tracking_graph.graph)), segmentation=pred_data.segmentation)

# CTC metrics on solution
# metrics_unchanged = run_metrics(
#     gt_data,
#     pred_data,
#     CTCMatched,
#     [CTCMetrics]
# )

# metrics_no_parents = run_metrics(
#     gt_data,
#     pred_data_no_parent_edges,
#     CTCMatched,
#     [CTCMetrics]
# )

# metrics_no_edges = run_metrics(
#     gt_data,
#     pred_data_no_edges,
#     CTCMatched,
#     [CTCMetrics]
# )
pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(metrics_unchanged)
# pp.pprint(metrics_no_parents)
# pp.pprint(metrics_no_edges)

div_unchanged = run_metrics(
    gt_data,
    pred_data,
    CTCMatched,
    [DivisionMetrics],
    metrics_kwargs={
        'frame_buffer': (0, )
    }
)
pp.pprint(div_unchanged)



