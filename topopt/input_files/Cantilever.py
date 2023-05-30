from dolfinx import mesh
from mpi4py import MPI
# if __name__ == "__main__":
#     from ProblemDef import TopOptProblem
# else:
#     from .ProblemDef import TopOptProblem
from topopt.input_files.ProblemDef import TopOptProblem
import numpy as np
from dolfinx import fem, mesh, io
from petsc4py import PETSc
import ufl
from typing import Callable
from dolfinx.fem.function import Constant


# mesh generation --------------------------------
class Cantilever2D(TopOptProblem):
    def __init__(self, isTest=False) -> None:
        super().__init__()
        self.dim = 2
        self.lx, self.ly = 160.,40.
        self.nelx, self.nely = (160, 40)
        if isTest:
            self.lx, self.ly = 10.,10
            self.nelx, self.nely = 10,10

        self.define_mesh()
        self.BC_loc, self.BC_u = self.define_BC()
        self.LC_loc, self.LC_force = self.define_LC()

    def define_dof(self):
        self.U1 = fem.VectorFunctionSpace(self.msh, ("CG", 1)) 
        self.D0 = fem.FunctionSpace(self.msh, ("DG", 0)) 

    def define_mesh(self):
        self.msh = mesh.create_rectangle(MPI.COMM_WORLD, (np.zeros(2), [self.lx, self.ly]), [self.nelx, self.nely], cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)

    def define_BC(self) -> (Callable, Constant):
        # define support ---------------------------------
        if MPI.COMM_WORLD.rank == 0:
            print("currently supports left clamp problem only")
        def left_clamp(x): 
            return np.isclose(x[0], 0.0)
        
        u_zero = np.array([0., 0.], dtype=PETSc.ScalarType) # predefined displacement
        return left_clamp, u_zero  

    def define_LC(self)  -> (list[tuple[int,Callable]], np.ndarray):
        if MPI.COMM_WORLD.rank == 0:
            print("currently supports point load only")
        load_points = [(1, lambda x: x[0] == self.nelx)]

        force = fem.Constant(self.msh, (0.0, -1.0)) 
        return load_points, force 
    
    def __str__(self) -> str:
        if MPI.COMM_WORLD.rank == 0:
            return f"Cantilever2D: lx={self.lx}, ly={self.ly}, nelx={self.nelx}, nely={self.nely}"
    
    
if __name__ == "__main__":
    # lx, ly = 160, 80
    # nelx, nely = 160, 80
    # mesh.create_rectangle(MPI.COMM_WORLD, (np.zeros(2), [lx, ly]), [nelx, nely])#, cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)
    cantilever = Cantilever2D()
    # print((type(cantilever.LC_loc[0][1]), type(cantilever.LC_force)))
    # print((type(cantilever.BC_loc), type(cantilever.BC_u)))