from itertools import count
import os

from tqdm import tqdm
from ctc_timings import get_im_centers
from flow_graph import FlowGraph
from visualize_lp_solution import get_gt_graph

DS_NAME = "Fluo-N2DL-HeLa/02_GT/"
OUTPUT_PATH = os.path.join("/home/draga/PhD/code/experiments/ctc/", DS_NAME, "output/")
SOLUTION_PATH = os.path.join(OUTPUT_PATH, "25Nov22_1332.sol")

GT_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME, 'TRA/')
IM_PATH = os.path.join("/home/draga/PhD/data/cell_tracking_challenge/", DS_NAME[:-4]+"/")

def load_lp_solution(sol_path, graph, coords, preserve_inconsistencies=False):
    migration_edges = set()
    with open(sol_path) as f:
        lines = f.readlines()

    # prep coords with 0 flows in and out
    coords['in-app'] = 0
    coords['in-div'] = 0
    coords['in-mig'] = 0
    coords['out-mig'] = 0
    coords['out-target'] = 0

    mig_info = ""
    graph._g.es['flow'] = [0 for _ in range(len(graph._g.es))]
    for line in tqdm(lines, total=len(lines), desc='Reading solution'):
        if line.startswith("e"):
            (var_name, flow) = line.strip().split()
            flow = float(flow)
            if flow > 0:
                current_edge = graph._g.es(var_name=var_name)[0]
                src = current_edge.source
                dest = current_edge.target
                if graph.MIGRATION_EDGE_REGEX.match(var_name):
                    if not preserve_inconsistencies:
                        # TODO: better handling of multiple parents
                        # handle merges by excluding edges when migration flow into the vertex would be >1
                        current_flow = coords.loc[[dest], ['in-mig']].values[0]
                        if current_flow + flow > 1:
                            mig_info += f'Migration flow into {dest} > 1 (from {src}). \n'
                            continue
                    migration_edges.add((src, dest))
                    coords.loc[[dest], ['in-mig']] += flow
                    coords.loc[[src], ['out-mig']] += flow
                    current_edge['flow'] = flow
                elif graph.DIVISION_EDGE_REGEX.match(var_name):
                    coords.loc[[dest], ['in-div']] += flow
                elif graph.APPEARANCE_EDGE_REGEX.match(var_name):
                    coords.loc[[dest], ['in-app']] += flow
                elif graph.EXIT_EDGE_REGEX.match(var_name):
                    coords.loc[[src], ['out-target']] += flow
                # else:
                #     print(f'Edge {var_name} matches no known regexes!')
    print(mig_info)
    return migration_edges

# # make FlowGraph from centers
# coords, min_t, max_t, corners = get_im_centers(GT_PATH)
# graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)
# # index in coords is index of vertex in graph
# gt_graph = set(get_gt_graph(coords, graph, GT_PATH))
# migration_edges = get_solution_edges(SOLUTION_PATH, graph)

# count_gt = len(gt_graph)
# count_ours = len(migration_edges)
# print('Num gt edges: ', count_gt)
# print('Num our edges: ', count_ours)

# tp = set()
# fp = set()
# fn = set()

# # true positives: in both graphs
# tp = migration_edges.intersection(gt_graph)
# # false positives: in ours but not theirs
# fp = migration_edges.difference(gt_graph)
# # false negatives: in theirs but not ours
# fn = gt_graph.difference(migration_edges)

# print('TP: ', len(tp))
# print('FP: ', len(fp))
# print('FN: ', len(fn))
