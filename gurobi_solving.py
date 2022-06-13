import gurobipy

# model = gurobipy.read('./close-together-model.lp')
model = gurobipy.read('./further-away-model.lp')
model.optimize()
model.printAttr('X')
