from contextlib import redirect_stdout
from datetime import datetime
import glob
import io
import os
import time

import napari
import numpy as np
from tqdm import tqdm
from flow_graph import FlowGraph
import pandas as pd
import gurobipy
from gurobipy import GRB
from ims_to_graph import load_tiff_frames, get_centers, get_point_coords

DATA_ROOT = '/home/draga/PhD/data/cell_tracking_challenge/'
OUT_ROOT = '/home/draga/PhD/code/experiments/ctc/'
DS_NAME = 'Fluo-N2DL-HeLa/'
SEQS = ['01_GT', '02_GT']
MIGRATION_ONLY = False

def get_im_centers(im_pth):
    im_arr = load_tiff_frames(im_pth)
    centers = get_centers(im_arr)
    center_coords = np.asarray(get_point_coords(centers))
    coords_df = pd.DataFrame(center_coords, columns=['t', 'y', 'x'])
    coords_df['t'] = coords_df['t'].astype(int)
    min_t = 0
    max_t = im_arr.shape[0]-1
    corners = [(0, 0), im_arr.shape[1:]]
    return coords_df, min_t, max_t, corners

def get_graph(coords, min_t, max_t, corners):
    start = time.time()
    graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t, migration_only=MIGRATION_ONLY)
    end = time.time()
    build_duration = end - start
    print("Build duration: ", build_duration)
    return graph, build_duration

def write_model(graph, model_path):
    start = time.time()
    if not os.path.exists(model_path):
        print("Writing model...")
        graph._to_lp(model_path)
        end = time.time()
        write_duration = end - start
        print("Write duration: ", write_duration)
    else:
        write_duration = 0
    return write_duration

def solve_model(model_path, sol_path):
    model = gurobipy.read(model_path)
    model.Params.TimeLimit = 300 # 5 minute timeout
    model.optimize()
    runtime = model.Runtime
    if model.status == GRB.OPTIMAL:
        model.write(sol_path)
    status = model.status
    return status, runtime

if __name__ == '__main__':
    for seq in SEQS:
        for _ in tqdm(range(50)):
            im_dir = os.path.join(DATA_ROOT, DS_NAME, seq, 'TRA/')
            model_root = os.path.join(OUT_ROOT, DS_NAME, seq, 'models/')
            sol_root = os.path.join(OUT_ROOT, DS_NAME, seq, 'output/')
            os.makedirs(model_root, exist_ok=True)
            os.makedirs(sol_root, exist_ok=True)

            current_datetime = datetime.now().strftime("%d%b%y_%H%M")
            out_path = os.path.join(OUT_ROOT, DS_NAME, seq, f'runtimes.csv')
            model_path = os.path.join(model_root, f'{current_datetime}.lp')
            sol_path = os.path.join(sol_root, f'{current_datetime}.sol')

            coords, min_t, max_t, corners = get_im_centers(im_dir)
            graph, build_time = get_graph(coords, min_t, max_t, corners)
            write_time = write_model(graph, model_path)
            solve_status, solve_time = solve_model(model_path, sol_path)
            if not os.path.exists(out_path):
                with open(out_path, 'w') as f:
                    header = 'ID,BUILD_MODEL,WRITE_MODEL,SOLVE_MODEL,SOLVE_STATUS,MIGRATION_ONLY\n'
                    f.write(header)
            with open(out_path, 'a') as f:
                info = f'{current_datetime},{build_time},{write_time},{solve_time},{solve_status},{int(MIGRATION_ONLY)}\n'
                f.write(info)

