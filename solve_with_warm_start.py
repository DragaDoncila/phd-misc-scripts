import gurobipy

model_path_no_div = ''
model_path_div = ''

model = gurobipy.read(model_path_no_div)
model.optimize()

vbasis = {}
# store bases 
for var in model.getVars():
    vbasis[var.VarName] =  var.VBasis

cbasis = []
for cstr in model.getConstrs():
    cbasis.append(cstr)

model_div = gurobipy.read(model_path_div)

# restore vbasis

# restore cbasis

# resolve
