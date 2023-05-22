import os
from datetime import datetime
from ctc_timings import get_im_centers, get_graph
from flow_graph import FlowGraph
from traccuracy.matchers._compute_overlap import get_labels_with_overlap
from visualize_lp_solution import load_tiff_frames
import networkx as nx
import igraph
import numpy as np
import pandas as pd

# load GT graph
def get_gt_graph(gt_path, return_ims=False):
    ims, coords, min_t, max_t, corners = get_im_centers(gt_path, return_ims = True)
    srcs = []
    dests = []
    is_parent = []
    for label_val in range(coords['label'].min(), coords['label'].max()):
        gt_points = coords[coords.label == label_val].sort_values(by='t')
        track_edges = [(gt_points.index.values[i], gt_points.index.values[i+1]) for i in range(0, len(gt_points)-1)]
        if len(track_edges):
            sources, targets = zip(*track_edges)
            srcs.extend(sources)
            dests.extend(targets)
            is_parent.extend([0 for _ in range(len(sources))])

    man_track = pd.read_csv(os.path.join(gt_path, 'man_track.txt'), sep=' ', header=None)
    man_track.columns = ['current', 'start_t', 'end_t', 'parent']
    child_tracks = man_track[man_track.parent != 0]
    for index, row in child_tracks.iterrows():
        parent_id = row['parent']
        parent_end_t = man_track[man_track.current == parent_id]['end_t'].values[0]
        parent_coords = coords[(coords.label == parent_id)][coords.t == parent_end_t]
        child_coords = coords[(coords.label == row['current']) & (coords.t == row['start_t'])]
        srcs.append(parent_coords.index.values[0])
        dests.append(child_coords.index.values[0])
        is_parent.append(1)

    edges = pd.DataFrame({
        'sources': srcs,
        'dests': dests,
        'is_parent': is_parent
    })    
    graph = igraph.Graph.DataFrame(edges, directed=True, vertices=coords, use_vids=True)
    if not return_ims:
        return graph, coords
    return ims, graph, coords

def store_solution_on_graph(opt_model, graph):
    sol_vars = opt_model.getVars()
    v_info = [v.VarName.lstrip('flow[').rstrip(']').split(',') + [v.X] for v in sol_vars]
    v_dict = {int(eid): {
        'var_name': var_name,
        'src_id': int(src_id),
        'target_id': int(target_id),
        'flow': float(flow)
    } for eid, var_name, src_id, src_label, target_id, target_label, flow in v_info if float(flow) > 0}

    # store the correct flow on each graph edge
    graph._g.es['flow'] = 0
    graph._g.es.select(list(v_dict.keys()))['flow'] = [v_dict[eid]['flow'] for eid in v_dict.keys()]

def get_gt_match_vertices(coords, gt_coords, sol_ims, gt_ims, v_id, label_key='label'):
    # get mask of problem blob
    problem_info = coords.loc[[v_id], [label_key, 't']]
    problem_label = problem_info[label_key].values[0]
    problem_t = problem_info['t'].values[0]
    if (ct := len(problem_info)) > 1:
        raise ValueError(f"Solution label {problem_label} appears {ct} times in frame {problem_t}.")
    # we're only interested in overlaps with this vertex
    only_problem_v_mask = sol_ims[problem_t] == problem_label
    gt_frame = gt_ims[problem_t]
    gt_ov_labels, _ = get_labels_with_overlap(gt_frame, only_problem_v_mask)
    gt_v_ids = []
    for label in gt_ov_labels:
        row = gt_coords[(gt_coords.label == label) & (gt_coords.t==problem_t)]
        if (ct := len(row)) > 1:
            raise ValueError(f"GT label {label} appears {ct} times in frame {problem_t}.")
        vid = row.index.values[0]
        gt_v_ids.append(vid)
    # some of these gt vertices might overlap with **other** vertices beyond this one
    # we need to filter those out.
    all_but_problem_v_mask = sol_ims[problem_t] * np.logical_not(only_problem_v_mask).astype(int)
    # we only found one gt vertex, or there's only one solution vertex in this frame
    if len(gt_v_ids) == 1 or all_but_problem_v_mask.max() == 0:
        return gt_v_ids
    real_overlaps = filter_other_overlaps(gt_v_ids, all_but_problem_v_mask, gt_frame)
    return real_overlaps

def filter_other_overlaps(gt_v_ids, sol_frame, gt_frame):
    real_overlaps = []
    for gt_v in gt_v_ids:
        v_label = gt_coords.loc[[gt_v], ['label']].values[0]
        only_gt_v_mask = gt_frame == v_label
        gt_overlaps, sol_overlaps = get_labels_with_overlap(only_gt_v_mask, sol_frame)
        # no bounding box overlaps, we can return 
        if not len(sol_overlaps):
            real_overlaps.append(gt_v)
        else:
            # check pixel by pixel overlaps
            if not has_overlapping_sol_vertex(sol_overlaps, gt_overlaps, only_gt_v_mask, sol_frame):
                real_overlaps.append(gt_v)
    return real_overlaps
                
def has_overlapping_sol_vertex(sol_overlaps, gt_overlaps, gt_frame, sol_frame):
    for i in range(len(gt_overlaps)):
        gt_label = gt_overlaps[i]
        sol_label = sol_overlaps[i]
        gt_blob = gt_frame == gt_label
        comp_blob = sol_frame == sol_label
        if blobs_intersect(gt_blob, comp_blob):
            return True
    return False

def blobs_intersect(gt_blob, comp_blob):
    intersection = np.logical_and(gt_blob, comp_blob) 
    return np.sum(intersection) > 0

def get_gt_unmatched_vertices_near_parent(coords, gt_coords, sol_ims, gt_ims, v_id, v_parents, dist, label_key='label'):
    from scipy.spatial import KDTree
    from traccuracy.matchers._compute_overlap import get_labels_with_overlap
    import numpy as np

    problem_row = coords.loc[[v_id]]
    problem_t = problem_row['t'].values[0]
    cols = ['y', 'x']
    if 'z' in coords.columns:
        cols = ['z', 'y', 'x']
    parent_rows = coords.loc[v_parents]
    parent_coords = parent_rows[cols].values
    
    # build kdt from gt frame
    gt_frame_coords = gt_coords[gt_coords['t'] == problem_t][cols]
    coord_indices, *coord_tuples = zip(*list(gt_frame_coords.itertuples(name=None)))
    coord_tuples = np.asarray(list(zip(*coord_tuples)))
    coord_indices = np.asarray(coord_indices)

    # get nearby vertices close to both parents of v
    gt_tree = KDTree(coord_tuples)
    nearby = [n_index for n_list in gt_tree.query_ball_point(parent_coords, dist, return_sorted=True) for n_index in n_list]
    potential_unmatched = coord_indices[nearby]
    unmatched = []
    problem_frame = sol_ims[problem_t]
    # check if they don't overlap with any solution vertices i.e. they are a fn
    for v in potential_unmatched:
        v_label = gt_coords.loc[[v], ['label']].values[0]
        mask = gt_ims[problem_t] == v_label
        _, sol_overlaps = get_labels_with_overlap(mask, problem_frame)
        if not len(sol_overlaps) and v not in unmatched:
            unmatched.append(v)
    return unmatched


def get_oracle(merge_node_ids, sol_graph, coords, gt_coords, sol_ims, gt_ims):
    last_label = 0
    last_index = 0
    v_info = None
    oracle = {}
    identified_gt_vs = set()
    for i in merge_node_ids:
        gt_matched = get_gt_match_vertices(coords, gt_coords, sol_ims, gt_ims, i)
        parent_ids = [v for v in sol_graph._g.neighbors(i, mode='in') if\
             sol_graph._g.es[sol_graph._g.get_eid(v, i)]['flow'] > 0 and\
                not sol_graph._is_virtual_node(sol_graph._g.vs[v])]
        gt_unmatched = get_gt_unmatched_vertices_near_parent(coords, gt_coords, sol_ims, gt_ims, i, parent_ids, 50)
        problem_v = coords.loc[[i]]
        problem_coords = tuple(problem_v[['y', 'x']].values[0])
        # we don't want to "reuse" a vertex we have already found
        gt_matched = list(filter(lambda v: v not in identified_gt_vs, gt_matched))
        gt_unmatched = list(filter(lambda v: v not in identified_gt_vs, gt_unmatched))            

        # we couldn't find a match for this vertex at all, we should just delete it
        if not len(gt_matched) and not len(gt_unmatched):
            decision = 'delete'
        # we've only found one vertex nearby, it's v itself
        elif len(gt_matched) + len(gt_unmatched) == 1:
            decision = 'terminate'
        # more than one "true" vertex overlaps v, a vertex should be introduced
        elif len(gt_matched) > 1:
            # closest match is `v`, second closest gets introduced
            distances_to_v = [np.linalg.norm(
                                np.asarray(problem_coords) - np.asarray(gt_coords.loc[[v], ['y', 'x']].values[0])
                            ) for v in gt_matched]
            second_closest = gt_matched[np.argsort(distances_to_v)[1]]
            v_info = gt_coords.loc[second_closest]
            decision = 'introduce'
            identified_gt_vs.add(second_closest)
        # we didn't find >1 overlap, but we've found an unmatched GT vertex nearby
        elif len(gt_unmatched):
            # we just take the closest
            v_id = gt_unmatched[0]
            v_info = gt_coords.loc[v_id]
            decision = 'introduce'
            identified_gt_vs.add(v_id)

        if v_info is not None:
            if last_label == 0:
                next_label = coords['label'].max() + 1
                # hypervertices...
                if max(coords.index.values) > sol_graph.division.index:
                    new_index = max(coords.index.values) + 1
                else: 
                    new_index = max(coords.index.values) + 5
            else:
                next_label = last_label + 1
                new_index = last_index + 1

            last_label = next_label
            last_index = new_index

        oracle[i] = {
            'decision': decision,
            'v_info': None if v_info is None else (int(new_index), list(v_info[['t', 'y', 'x']]) + [int(next_label)]),
            'parent': None
        }
        v_info = None
    return oracle

def store_flow(nx_sol, ig_sol, hyper_mapping=None):
    ig_sol._g.es.set_attribute_values('flow', 0)
    flow_es = nx.get_edge_attributes(nx_sol, 'flow')
    for e_id, flow in flow_es.items():
        src, target = e_id
        if hyper_mapping is not None and src in hyper_mapping:
            hyper_v = hyper_mapping[src]
            src = ig_sol.getattr(hyper_v).index
        if hyper_mapping is not None and target in hyper_mapping:
            hyper_v = hyper_mapping[target]
            target = ig_sol.getattr(hyper_v).index
        ig_sol._g.es[ig_sol._g.get_eid(src, target)]['flow'] = flow
        
def load_sol_data(sol_pth, seg_pth):
    sol = nx.read_graphml(sol_pth, node_type=int)
    sol_ims = load_tiff_frames(seg_pth)
    oracle_node_df = pd.DataFrame.from_dict(sol.nodes, orient='index')
    oracle_node_df.rename(columns={'pixel_value':'label'}, inplace=True)
    oracle_node_df.drop(oracle_node_df.tail(4).index, inplace = True)
    im_dim =  [(0, 0), sol_ims.shape[1:]]
    min_t = 0
    max_t = sol_ims.shape[0] - 1
    sol_g = FlowGraph(im_dim, oracle_node_df, min_t, max_t)
    store_flow(sol, sol_g)
    return sol_g, sol_ims, oracle_node_df

def load_final_sol_data(sol_pth, seg_pth):
    sol = nx.read_graphml(sol_pth, node_type=int)
    sol_ims = load_tiff_frames(seg_pth)
    im_dim =  [(0, 0), sol_ims.shape[1:]]
    min_t = 0
    max_t = sol_ims.shape[0] - 1
    oracle_node_df = pd.DataFrame.from_dict(sol.nodes, orient='index')
    oracle_node_df.rename(columns={'pixel_value':'label'}, inplace=True)
    hyper_indices = list(oracle_node_df[(oracle_node_df['t'] < 0) | (oracle_node_df['t'] > max_t)].index)
    hyper_mappping = {}
    for idx in hyper_indices:
        if oracle_node_df.loc[idx, 'is_appearance']:
            hyper_mappping[idx] = 'appearance'
        elif oracle_node_df.loc[idx, 'is_division']:
            hyper_mappping[idx] = 'division'
        elif oracle_node_df.loc[idx, 'is_target']:
            hyper_mappping[idx] = 'target'
        elif oracle_node_df.loc[idx, 'is_source']:
            hyper_mappping[idx] = 'source'
    oracle_node_df = oracle_node_df.drop(index=hyper_indices)
    sol_g = FlowGraph(im_dim, oracle_node_df, min_t, max_t)
    store_flow(sol, sol_g, hyper_mappping)
    return sol_g, sol_ims, oracle_node_df

def get_merges(sol_g):
    merges = []
    for v in sol_g._g.vs:
        if not (v['is_appearance'] or v['is_division'] or v['is_target']):
            neighbours = sol_g._g.neighbors(v, 'in')
            real_neighbours = []
            for n in neighbours:
                nv = sol_g._g.vs[n]
                if not (nv['is_appearance'] or nv['is_division']):
                    if sol_g._g.es[sol_g._g.get_eid(n, v)]['flow'] > 0:
                        real_neighbours.append(n)
            if len(real_neighbours) > 1:
                merges.append(v.index)
    return merges

def mask_new_vertices(introduce_info, sol_ims, gt_ims):
    for intro_t, coords, new_label in introduce_info.values():
        gt_frame = gt_ims[intro_t]
        int_coords = tuple(int(coord) for coord in coords)
        mask = gt_frame == gt_frame[int_coords]
        sol_ims[intro_t][mask] = new_label

def terminate_vertices(vertices, sol_g):
    get_distance = lambda x, y: abs(np.linalg.norm(np.asarray(x['coords'])) - np.linalg.norm(np.asarray(y['coords'])))
    actually_changed = []
    for vertex in vertices:
        neighbours = sol_g._g.neighbors(vertex, 'in')
        real_neighbours = []
        for n in neighbours:
            nv = sol_g._g.vs[n]
            if not (nv['is_appearance'] or nv['is_division']):
                if sol_g._g.es[sol_g._g.get_eid(n, vertex)]['flow'] > 0 and\
                    sol_g._g.es[sol_g._g.get_eid(n, sol_g.target)]['cost'] != 0:
                    # in case we've already divested this parent
                    real_neighbours.append(nv)
        # if this merge vertex has already been resolved, we don't mess with anything
        if len(real_neighbours) > 1:
            furthest_parent = real_neighbours[0]
            for parent in real_neighbours:
                if get_distance(parent, vertex) > get_distance(furthest_parent, vertex):
                    furthest_parent = parent
            # make edge furthest parent, target cost = 0
            sol_g._g.es[sol_g._g.get_eid(furthest_parent, sol_g.target)]['cost'] = 0
            actually_changed.append(vertex)
    return actually_changed


if __name__ == '__main__':
    import time
    import json

    ROOT_DATA_DIR = '/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/'
    ALL_DATASETS = [os.path.basename(pth) for pth in os.listdir(ROOT_DATA_DIR) if not pth.endswith('.csv')]
    ds_summary_df = pd.read_csv('/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/ds_summary.csv', index_col=0)
    with open('/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/ctc_metrics.json', 'r') as f:
        metric_info = json.load(f)
    with open('/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/merge_info.json', 'r') as f:
        merges = json.load(f)
    oracle_pth = '/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/oracles.json'
    with open(oracle_pth, 'r') as f:
        oracles = json.load(f)

    ds_summary_df = ds_summary_df.sort_values(by='n_frames')
    used_ds_names = []
    rebuild_durations = []
    rebuilt_frames  = []
    total_merges = []
    total_introduced = []
    
    for i, row in enumerate(ds_summary_df.itertuples(), 0):
    # for key in ['PhC-C2DL-PSC_1', 'BF-C2DL-MuSC_2']:
        ds_name = row.ds_name
        seq = row.seq
        key = f'{ds_name}_{seq}'
        
        # if (key in oracles and oracles[key] is not None) or\
        #     key in ['PhC-C2DL-PSC_1', 'BF-C2DL-MuSC_2']:
        #     continue
        # ds_name, seq = tuple(key.split('_'))
        # seq = int(seq)

        sol_dir = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_RES/'.format(seq))
        sol_pth = os.path.join(sol_dir, 'final_solution.graphml')
        seg_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_ST/SEG/'.format(seq))
        gt_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_GT/TRA/'.format(seq))

        if key in oracles and oracles[key] is not None:
            final_oracle = oracles[key]

            print(f"Running oracle for {key}")
            sol_g, sol_ims, sol_coords = load_final_sol_data(sol_pth, seg_pth)
            # gt_ims, gt_graph, gt_coords = get_gt_graph(gt_pth, return_ims=True)
            
            actually_changed = []
            term_vertices = [int(key) for key in final_oracle if final_oracle[key]['decision'] == 'terminate']
            terminate_ts = list(sorted(set([sol_g._g.vs[int(node)]['t'] for node in term_vertices])))
            for t in terminate_ts:
                frame_vertices = [sol_g._g.vs[k] for k in term_vertices if sol_g._g.vs[k]['t']==t]
                actually_changed.extend(terminate_vertices(frame_vertices, sol_g))

            # re-build and solve model
            m, flow = sol_g._to_gurobi_model()
            m.optimize()
            if m.Status == 3:
                infinite_cost_edges = sol_g._g.es.select(cost_ge=1e10)
                sol_g._g.delete_edges(infinite_cost_edges)
                m, flow = sol_g._to_gurobi_model()
                m.optimize()
                if m.Status != 2:
                    raise ValueError(f"Attempted to remove infinite cost edges but model for {key} was still not solved.")
            
            # save new solution onto graph 
            store_time = sol_g.store_solution(m)
            sol_coords = sol_g.get_coords_df()
            
            current_merges = get_merges(sol_g)
            new_merges = set(current_merges) - (set(final_oracle.keys()) - set(term_vertices))
                            
            final_sol_pth = os.path.join(sol_dir, 'terminate_solution.graphml')
            nx_g = sol_g.convert_sol_igraph_to_nx()
            nx.write_graphml_lxml(nx_g, final_sol_pth)
            
            # used_ds_names.append(key)
            # rebuild_durations.append(rebuild_duration)
            # rebuilt_frames.append(n_rebuilt)
            # total_merges.append(len(oracle))
            # total_introduced.append(len([k for k in oracle if oracle[k]['decision'] == 'introduce']))
        # for i in range(len(used_ds_names)):
        #     ds_name = used_ds_names[i]
        #     name, seq = tuple(ds_name.split('_'))
        #     row_idx = ds_summary_df[(ds_summary_df.ds_name == name) & (ds_summary_df.seq == int(seq))].index[0]
        #     ds_summary_df.at[row_idx,'rebuild_time']= rebuild_durations[i]
        #     ds_summary_df.at[row_idx, 'n_rebuilt'] = rebuilt_frames[i]
        #     ds_summary_df.at[row_idx, 'total_merges'] = total_merges[i]
        #     ds_summary_df.at[row_idx, 'total_introduced'] = total_introduced[i]
        # ds_summary_df.to_csv('/media/ddon0001/Elements/BMVC/data/cell_tracking_challenge/ST_Segmentations/ds_summary.csv')
