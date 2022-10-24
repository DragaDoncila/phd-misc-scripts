from datetime import datetime
import os
import time
from flow_graph import FlowGraph
import pandas as pd
import gurobipy

DS_NAME = 'Fluo-N2DL-HeLa/01/'
CENTERS_PATH = os.path.join('/home/draga/PhD/code/repos/misc-scripts/ctc/', DS_NAME, 'centers.csv')

OUT_ROOT = '/home/draga/PhD/code/experiments/ctc/'
MODEL_ROOT = os.path.join(OUT_ROOT, DS_NAME, 'models/')
SOL_ROOT = os.path.join(OUT_ROOT, DS_NAME, 'output/')
os.makedirs(MODEL_ROOT, exist_ok=True)
os.makedirs(SOL_ROOT, exist_ok=True)

current_datetime = datetime.now().strftime("%d%b%y_%H%M")
out_path = os.path.join(OUT_ROOT, DS_NAME, f'runtimes.csv')
model_path = os.path.join(MODEL_ROOT, f'{current_datetime}.lp')
sol_path = os.path.join(SOL_ROOT, f'{current_datetime}.sol')

# open centers
node_df = pd.read_csv(CENTERS_PATH)
coords = node_df[['t', 'y', 'x']]
min_t = 0
max_t = coords['t'].max()
corners = [(0, 0), (1024, 1024)]

# load into flow graph
start = time.time()
graph = FlowGraph(corners, coords, min_t=min_t, max_t=max_t)
end = time.time()
build_duration = end - start
print("Build duration: ", build_duration)

if not os.path.exists(model_path):
    print("Writing model...")
    graph._to_lp(model_path)
    end2 = time.time()
    write_duration = end2 - end
    print("Write duration: ", write_duration)
else:
    write_duration = 0

print("Solving model...")
model = gurobipy.read(model_path)
model.optimize()
# model.printAttr('X')
solve_duration = model.Runtime
model.write(sol_path)
# solve_duration = 0

if not os.path.exists(out_path):
    with open(out_path, 'w') as f:
        header = 'ID,BUILD_MODEL,WRITE_MODEL,SOLVE_MODEL\n'
        f.write(header)
with open(out_path, 'a') as f:
    info = f'{current_datetime},{build_duration},{write_duration},{solve_duration}\n'
    f.write(info)
