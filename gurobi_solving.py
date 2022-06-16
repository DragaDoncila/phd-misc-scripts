import gurobipy

# model = gurobipy.read('./close-together-model.lp')
model = gurobipy.read('./cell_swaps.lp')
model.optimize()
model.printAttr('X')
