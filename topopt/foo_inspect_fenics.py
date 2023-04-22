'''
    this script shows types of fenics objects
    and how to inspect them
'''

import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from util import *
from petsc4py import PETSc


msh = mesh.create_box(MPI.COMM_WORLD, [[0,0,0],[1,1,1]], [2,2,2], cell_type=mesh.CellType.hexahedron)
print(type(msh)) # msh.Mesh
dim = msh.topology.dim
print(msh.topology.dim)

print(msh.topology.index_map(dim).size_local) # number of cells in local process
print(msh.topology.index_map(dim).size_global) # number of cells in global process


U1 = fem.VectorFunctionSpace(msh, ("CG", 1)) # displacement basiss
u, v = ufl.TrialFunction(U1), ufl.TestFunction(U1)


C1 = fem.FunctionSpace(msh, ("CG", 1))
D0 = fem.FunctionSpace(msh, ("DG", 0))
u_sol, density_old, density = fem.Function(U1), fem.Function(D0), fem.Function(D0)
den_node, den_sens = fem.Function(C1), fem.Function(C1)

print(f" type of u: {type(u)}, type of v: {type(v)}, type of u_sol: {type(u_sol)}")
#  type of u: <class 'ufl.argument.Argument'>, type of v: <class 'ufl.argument.Argument'>, type of u_sol: <class 'dolfinx.fem.function.Function'>

print(f"type of density: {type(density)}, type of den_node: {type(den_node)}, type of den_sens: {type(den_sens)}, type of density.array: {type(density.x.array)}")
# type of density: <class 'dolfinx.fem.function.Function'>, type of den_node: <class 'dolfinx.fem.function.Function'>, type of den_sens: <class 'dolfinx.fem.function.Function'>

print(f"type of PetscScalar: {type(PETSc.ScalarType(0.5))}")