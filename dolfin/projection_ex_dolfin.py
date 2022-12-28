# Set python packages
import numpy as np
from dolfin import *
from mpi4py import MPI

# Declare the input variables
nelx = 2
nely = 2

# Put MPI commands
comm = MPI.COMM_WORLD

# Prepare finite element analysis
msh = RectangleMesh(Point(0, 0), Point(nelx, nely), nelx, nely, "right/left")
U = VectorFunctionSpace(msh, "P", 1)
D = FunctionSpace(msh, "DG", 0)
u, d = Function(U), Function(D)

# Initial input value
u.vector()[:] = 0.1
print(f"Initial value on process {MPI.COMM_WORLD.rank}: \n{u.vector()[:]}\n")

# Projection CG-1 to DG-0
proj = project(sum(u), D)
print(f"Projection psi value on process {MPI.COMM_WORLD.rank}: \n{proj.vector()[:]}\n")