# PYTHON LIBRARIES ---------------------------------
import ufl
from dolfinx import fem

# DOLFINX PROJECTION FUNCTION ---------------------------------
'''
input:
    dfx_func (dolfinx.fem.Function): function which want to be projected other function space
    func_space (dolfinx.fem.FunctionSpace): function space user want to project some other dimension

output:
    result_sol (dolfinx.fem.Function): projected function which dependent func_space
'''
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    project_prob = fem.petsc.LinearProblem(a, l, [])
    result_sol = project_prob.solve()
    return result_sol