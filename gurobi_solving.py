import gurobipy

# model = gurobipy.read('./close-together-model.lp')
model = gurobipy.read('./cell_swaps_autogen2.lp')
model.optimize()
model.printAttr('X')
