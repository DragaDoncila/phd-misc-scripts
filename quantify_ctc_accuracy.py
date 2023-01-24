from itertools import count
import os
from ctc_timings import get_im_centers
from flow_graph import FlowGraph
from visualize_lp_solution import get_gt_graph

DS_NAME = "Fluo-N2DL-HeLa/02_GT/"
OUTPUT_PATH = os.path.join("/home/draga/PhD/code/experiments/ctc/", DS_NAME, "output/")
SOLUTION_PATH = os.path.join(OUTPUT_PATH, "25Nov22_1332.sol")

GT_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME, 'TRA/')
IM_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME[:-4]+"/")

# make FlowGraph from centers
coords, min_t, max_t, corners = get_im_centers(GT_PATH)
graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)
# index in coords is index of vertex in graph
gt_graph = set(get_gt_graph(coords, graph, GT_PATH))

migration_edges = set()
with open(SOLUTION_PATH) as f:
    for line in f:
        if line.startswith("e"):
            (var_name, flow) = line.strip().split()
            flow = float(flow)
            if flow > 0:
                if graph.MIGRATION_EDGE_REGEX.match(var_name):
                    current_edge = graph._g.es(var_name=var_name)[0]
                    migration_edges.add((current_edge.source, current_edge.target))

count_gt = len(gt_graph)
count_ours = len(migration_edges)
print('Num gt edges: ', count_gt)
print('Num our edges: ', count_ours)

tp = set()
fp = set()
fn = set()

# true positives: in both graphs
tp = migration_edges.intersection(gt_graph)
# false positives: in ours but not theirs
fp = migration_edges.difference(gt_graph)
# false negatives: in theirs but not ours
fn = gt_graph.difference(migration_edges)

print('TP: ', len(tp))
print('FP: ', len(fp))
print('FN: ', len(fn))
