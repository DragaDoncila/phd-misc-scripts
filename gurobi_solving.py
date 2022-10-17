import gurobipy

# model = gurobipy.read('./close-together-model.lp')
model = gurobipy.read('/home/draga/PhD/code/experiments/140521_late_te_all_tracks/models/10-Oct-22_18:07.lp')
model.Params.method = 1
model.Params.presolve  = 2
model.optimize()
model.printAttr('X')
model.write('/home/draga/PhD/code/experiments/140521_late_te_all_tracks/output/10-Oct-22_18:07.sol')
