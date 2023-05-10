import json
import pandas as pd
import os
from bmvc_metrics import load_sol


ROOT_DATA_DIR = '/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/'
ds_summary_df = pd.read_csv('/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/ds_summary.csv', index_col=0)

with open('/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/merge_info.json', 'r') as f:
    merges = json.load(f)
    
for i, row in enumerate(ds_summary_df.itertuples(), 1):
    ds_name = row.ds_name
    seq = row.seq
    key = f'{ds_name}_{seq}'
    if key in merges:
        print(f'Skipping {key}, merges already counted.')
        continue
    else:
        merges[key] = {}
    
    sol_dir = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_RES/'.format(seq))
    sol_pth = os.path.join(sol_dir, 'full_solution.graphml')
    seg_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_ST/SEG/'.format(seq))
    gt_pth = os.path.join(ROOT_DATA_DIR, ds_name, '{0:02}_GT/TRA/'.format(seq))
    
    # load solution, check the number of merges
    if not os.path.exists(sol_pth):
        merges[key]['n_merges'] = -1
        merges[key]['merge_nodes'] = []
    else:
        sol_data = load_sol(sol_pth, seg_pth)
        sol = sol_data.tracking_graph.graph
        oracle_node_df = pd.DataFrame.from_dict(sol.nodes, orient='index')
        
        merge_nodes = [node for node in sol.nodes if len(sol.in_edges(node)) > 1]
        merges[key]['n_merges'] = len(merge_nodes)
        merges[key]['merge_nodes'] = merge_nodes
        print(f"{key}: {merges[key]['n_merges']}")
        with open('/home/draga/PhD/data/cell_tracking_challenge/ST_Segmentations/merge_info.json', 'w') as f:
            json.dump(merges, f)
    
