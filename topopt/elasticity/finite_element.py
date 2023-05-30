import ufl
import numpy as np
# if __name__ == "__main__":
#     from material import *
# else:
#     from .material import *
# from .material import MaterialModels  
import sys
sys.path.append("../")
from topopt.input_files import ProblemDef
from topopt.input_files.Cantilever import Cantilever2D
from topopt.elasticity.material import *
from topopt.elasticity.material import Material_el_lin_iso
# from input_files.Cantilever import Cantilever2D
# from input_files import ProblemDef
# from .material import *
from dolfinx import mesh, fem
from mpi4py import MPI

from abc import *
class FiniteElement(ABC):
    def __init__(self, msh:mesh, mat:MaterialModels) -> None:
        super().__init__()
        self.msh = msh
        self.mat = mat
    @abstractmethod
    def applyBC(self):
        pass    
    def assemble_without_solve(self):
        A_foo = fem.petsc.create_matrix(self.FE._a)
        A_foo.zeroEntries()
        fem.petsc._assemble_matrix_mat(A_foo, self.FE._a, bcs=[]) 
        A_foo.assemble()
        self.FE._A0 = A_foo

    def solve(self):
        try:
            self.u_sol = self.FE.solve()
        except:
            print("solve failed")
            pass

    def get_global_stiffness(self) -> np.ndarray:
        return self.FE.A

class FE_LinMech(FiniteElement):
    # define macro

    def __init__(self, prob:ProblemDef, mat:MaterialModels) -> None:
        super().__init__(prob.msh, mat)
        self.prob = prob
        self.isAssembled = False
        if self.prob.dim == 2:
            if MPI.COMM_WORLD.rank == 0:
                print(f"{self.prob.dim} dimensional problem, and mesh size of {prob.nelx} x {prob.nely} is created")
        elif self.prob.dim == 3:
            if MPI.COMM_WORLD.rank == 0:
                print(f"{self.prob.dim} dimensional problem, and mesh size of {prob.nelx} x {prob.nely} x {prob.nelz} is created")
        self.setFunctionSpace()
        if MPI.COMM_WORLD.rank == 0:
            print("applying BC ...")
        self.applyBC()
        if MPI.COMM_WORLD.rank == 0:
            print("applying LC ...")
        self.applyLC() 
        if MPI.COMM_WORLD.rank == 0:
            print("done.\nsetting problem ...")
        self.setProblem()
        if MPI.COMM_WORLD.rank == 0:
            print("done")

    def setFunctionSpace(self):
        # function spaces ----------------------------
        self.U1 = fem.VectorFunctionSpace(self.msh, ("CG", 1)) # displacement basiss
        self.D0 = fem.FunctionSpace(self.msh, ("DG", 0)) # density
        self.u, self.v = ufl.TrialFunction(self.U1), ufl.TestFunction(self.U1) # note that this is ufl
        self.u_sol = fem.Function(self.U1) 

    def applyBC(self):
        f_dim = self.msh.topology.dim - 1 # facet dimension
        bc_facets = mesh.locate_entities_boundary(self.msh, f_dim, self.prob.BC_loc) # boundary facets
        bc_dofs   = fem.locate_dofs_topological(self.U1, f_dim, bc_facets) # boundary dofs
        bc_l = fem.dirichletbc(self.prob.BC_u, bc_dofs, self.U1)
        self.bcs = [bc_l]

    def applyLC(self):
        facet_indices, facet_markers = [], []
        f_dim = self.msh.topology.dim - 1
        for (marker, locator) in self.prob.LC_loc:
            facets = mesh.locate_entities(self.msh, f_dim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(self.msh, f_dim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        ds = ufl.Measure("ds", domain=self.msh, subdomain_data=facet_tag) # measure for facet (surface)
        self.f = ufl.dot(self.v, self.prob.LC_force) * ds(1) 

    def setProblem(self):
        def epsilon(_u):
            return ufl.sym(ufl.grad(_u))
        def sigma(_u, lmd, mu):
            return 2.0 * mu * epsilon(_u) + lmd * ufl.tr(epsilon(_u)) * ufl.Identity(len(_u))
        k = ufl.inner(sigma(self.u, self.mat.lmd, self.mat.mu), ufl.grad(self.v)) * ufl.dx
        petsc_options={"ksp_type": "preonly","pc_type": "lu","pc_factor_mat_solver_type": "mumps"}
        self.FE = fem.petsc.LinearProblem(k, self.f, bcs=self.bcs, petsc_options=petsc_options)
    
if __name__ == "__main__":
    cant = Cantilever2D()
    mat = Material_el_lin_iso()
    fe = FE_LinMech(cant, mat)
