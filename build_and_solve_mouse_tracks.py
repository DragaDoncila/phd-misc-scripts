from os import path
from re import A
import sys
import time
import dask.dataframe as df
from flow_graph import FlowGraph
import gurobipy
from datetime import datetime

ALL_TRACKS = '/home/draga/PhD/code/repos/misc-scripts/140521_late/140521_late_te_all_tracks_napari.csv'
MIGRATION_ONLY = False
MODEL_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability/models/'
SOL_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability/output/'
OUT_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability'

tracks_data = df.read_csv(ALL_TRACKS)
tracks_coords = tracks_data[['t', 'z', 'y', 'x']]
max_t = tracks_coords['t'].max().compute()
min_t = tracks_coords['t'].min().compute()
im_corners = [(505, 10, 713), (4340, 1964, 2157)]

for n_frames in [3, 4, 5, 7, 8, 9]:
    temp_max = min_t + n_frames
    print(f"Building {n_frames} frames...")
    needed_coords = tracks_coords[tracks_coords['t'] <= temp_max].reset_index(drop=True)
    current_datetime = datetime.now().strftime("%d%b%y_%H%M")
    out_path = path.join(OUT_PATH, f'runtimes-build-write.csv')
    model_id = f'{current_datetime}_{n_frames}{"_div" if not MIGRATION_ONLY else ""}'
    model_path = path.join(MODEL_PATH, f'{model_id}.lp')
    # sol_path = path.join(SOL_PATH, f'{current_datetime}.sol')
    start = time.time()
    graph = FlowGraph(im_corners, needed_coords, min_t=min_t, max_t=temp_max, migration_only=MIGRATION_ONLY)
    end = time.time()
    build_duration = end - start
    print("Build duration: ", build_duration)

    if not path.exists(model_path):
        print("Writing model...")
        graph._to_lp(model_path)
        end2 = time.time()
        write_duration = end2 - end
        print("Write duration: ", write_duration)
    else:
        write_duration = 0

    if not path.exists(out_path):
        with open(out_path, 'w') as f:
            header = 'ID,BUILD_MODEL,WRITE_MODEL,NFRAMES,MIGRATION_ONLY\n'
            f.write(header)
    with open(out_path, 'a') as f:
        info = f'{model_id},{build_duration},{write_duration},{n_frames},{int(MIGRATION_ONLY)}\n'
        f.write(info)

# print("Solving model...")
# model = gurobipy.read(model_path)
# model.Params.method = 1
# model.Params.presolve  = 2
# model.optimize()
# # model.printAttr('X')
# solve_duration = model.Runtime
# model.write(sol_path)
# solve_duration = 0

# if not path.exists(out_path):
#     with open(out_path, 'w') as f:
#         header = 'ID,BUILD_MODEL,WRITE_MODEL,SOLVE_MODEL,MIGRATION_ONLY\n'
#         f.write(header)
# with open(out_path, 'a') as f:
#     info = f'{current_datetime},{build_duration},{write_duration},{solve_duration},{int(MIGRATION_ONLY)}\n'
#     f.write(info)
