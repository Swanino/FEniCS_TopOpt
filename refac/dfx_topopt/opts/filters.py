# PYTHON LIBRARIES ---------------------------------
import ufl
from dolfinx import fem

# HELMHOLTZ WEAK FORM PDE FILTER FUNCTION ---------------------------------
'''
    helmholtz partial differential equation filter
    inputs: 
        rho_n(dolfinx.fem.Function): C1 space function to be determined
        r_min: helmholtz PDE filter radius (This value must be at least greater than 3.0)
    output:
        rho(dolfinx.fem.Function): determined value of C1 space function
'''
def helm_filter(rho_n, r_min):
    V = rho_n.ufl_function_space()
    rho, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (r_min**2) * ufl.inner(ufl.grad(rho), ufl.grad(w)) * ufl.dx + rho * w * ufl.dx
    L = rho_n * w * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, [])
    rho = problem.solve()
    return rho