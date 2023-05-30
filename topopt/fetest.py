from elasticity import material, finite_element
from input_files import Cantilever
from petsc4py import PETSc
from mpi4py import MPI  
import time
import numpy as np

def fetest():
    t = time.time()
    cant = Cantilever.Cantilever2D(isTest=True)
    mat =  material.Material_el_lin_iso()
    fe = finite_element.FE_LinMech(cant, mat)
    fe.setProblem()
    fe.assemble_without_solve()
    fe.solve()

    viewer = PETSc.Viewer(comm=MPI.COMM_WORLD)
    fe.u_sol.vector.view(viewer.ASCII(f"./savem/u4.vtu"))

    unorm = fe.u_sol.x.norm()
    usum = fe.u_sol.vector.sum()

    if MPI.COMM_WORLD.rank == 0:
        print("Solution vector norm:", unorm)
        print("Solution vector sum:", usum)

    if MPI.COMM_WORLD.rank == 0:
        print(f"Time taken: {time.time()-t}")

def toptest():
    pass

if __name__ == "__main__":
    fetest()