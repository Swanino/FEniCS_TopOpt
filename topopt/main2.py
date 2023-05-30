from elasticity import material, finite_element
from input_files import Cantilever
from petsc4py import PETSc
from mpi4py import MPI  

if __name__ == "__main__":
    cant = Cantilever.Cantilever2D()
    mat =  material.Material_el_lin_iso()
    fe = finite_element.FE_LinMech(cant, mat)
    fe.setProblem()
    fe.assemble_without_solve()
    fe.solve()

    viewer = PETSc.Viewer(comm=MPI.COMM_WORLD)
    # print(fe.u_sol.x.array[:])
    # viewer
    fe.u_sol.vector.view(viewer.ASCII("./savem/U.vtu"))
    fe.FE._A0.view(viewer.ASCII("./savem/testK0.vtu"))
    A = fe.get_global_stiffness()
    A.view(viewer.ASCII("./savem/testK.vtu"))

