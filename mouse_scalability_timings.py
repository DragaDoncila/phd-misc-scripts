import glob
import os
import sys
import gurobipy
from gurobipy import GRB


MODEL_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability/models/'
SOL_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability/output/'
OUT_PATH = '/home/draga/PhD/code/experiments/140521_late_te_all_tracks/scalability/runtimes-solve.csv'
all_models = sorted(list(glob.glob(f'{MODEL_PATH}/*.lp')))

# for each model
for model_path in all_models:
    for method in [2, 1]:
        model_id = os.path.basename(model_path)[:-3]
        n_frames = model_id.split('_')[2]
        migration_only = not 'div' in model_id

        sol_path = os.path.join(SOL_PATH, f'{model_id}.sol')

        with gurobipy.Env('', params={'MemLimit': 12}) as env:
            print("Solving model...")
            model = gurobipy.read(model_path)
            model.Params.method = method
            model.Params.TimeLimit = 300 # 5 minute timeout
            try:
                model.optimize()
            except Exception as e:
                continue
            # model.printAttr('X')
            if model.status == GRB.OPTIMAL:
                model.write(sol_path)
            status = model.status
            solve_duration = model.Runtime

            if not os.path.exists(OUT_PATH):
                with open(OUT_PATH, 'w') as f:
                    header = 'ID,SOLVE_MODEL,SOLVE_STATUS,SOLVE_METHOD,MIGRATION_ONLY\n'
                    f.write(header)
            with open(OUT_PATH, 'a') as f:
                info = f'{model_id},{solve_duration},{status},{method},{int(migration_only)}\n'
                f.write(info)
