import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt

# HELMHOLTZ WEAK FORM PDE FILTER FUNCTION ---------------------------------
def helm_filter(rho_n, r_min):
    V = rho_n.ufl_function_space()
    rho, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (r_min**2) * ufl.inner(ufl.grad(rho), ufl.grad(w)) * ufl.dx + rho * w * ufl.dx
    L = rho_n * w * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, [])
    rho = problem.solve()
    return rho

# DOLFINX PROJECTION FUNCTION ---------------------------------
'''
input: 
    dfx_func (dolfinx.function.Function): function to be projected
    func_space (dolfinx.function.FunctionSpace): function space to project to
'''
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    project_prob = fem.petsc.LinearProblem(a, l, [])
    result_sol = project_prob.solve()
    return result_sol

# PRINT MESH TO XDMF FILE ---------------------------------
def print_mesh(mesh, filename:str="output/density.xdmf") -> str:
    with io.XDMFFile(MPI.COMM_WORLD, filename, "w") as xdmf:
        xdmf.write_mesh(mesh)
    return filename

# project to Gauss quadrature points
def project_to_quadpts(dfx_func, quadpts):
    # project to quad points
    quadpts_func = fem.Function(quadpts.function_space)
    quadpts_func.vector.set(dfx_func.compute_point_values(quadpts.points))
    return quadpts_func