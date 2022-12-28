# Set python packages
import ufl
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

# Dolfinx projection function
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    project_prob = fem.petsc.LinearProblem(a, l, [])
    result_sol = project_prob.solve()
    return result_sol

# Declare the input variables
nelx = 2
nely = 2

# Put MPI commands
comm = MPI.COMM_WORLD

# Prepare finite element analysis
msh = mesh.create_rectangle(comm, ((0.0, 0.0), (nelx, nely)), (nelx, nely), cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right_left)
# msh = mesh.create_rectangle(comm, ((0.0, 0.0), (nelx, nely)), (nelx, nely), cell_type=mesh.CellType.quadrilateral)
U = fem.VectorFunctionSpace(msh, ("CG", 1))
D = fem.FunctionSpace(msh, ("DG", 0))
u, d = fem.Function(U), fem.Function(D)

# Initial input value
u.x.array[0] = 1
u.x.array[6] = 1
print(f"Initial value on process {MPI.COMM_WORLD.rank}: \n{u.x.array[:]}\n")

# Projection CG-1 to DG-0
proj = project_func(sum(u), D)
print(f"Projection psi value on process {MPI.COMM_WORLD.rank}: \n{proj.x.array[:]}\n")