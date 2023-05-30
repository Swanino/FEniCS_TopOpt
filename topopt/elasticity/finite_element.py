import ufl
import numpy as np
from elasticity import *
from elasticity.material import MaterialModels  
from input_files.cantilever import Cantilever2D
from input_files import TopOptProblem
from material import *
from dolfinx import mesh, fem

from abc import *
class FiniteElement(ABC):
    def __init__(self, msh:mesh, mat:MaterialModels) -> None:
        super().__init__()
        self.msh = msh
        self.mat = mat
    @abstractmethod
    def get_local_stiffness(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_global_stiffness(self):
        pass
    @abstractmethod
    def applyBC(self):
        pass
    


class FE_LinMech(FiniteElement):
    # define macro

    def __init__(self, prob:TopOptProblem, mat:MaterialModels) -> None:
        super().__init__(prob.msh, mat)
        self.prob = prob
        self.isAssembled = False
        print(f"{self.dim} dimensional problem, and mesh size of {prob.nelx} x {prob.nely} x {prob.nelz} is created")

    def setFunctionSpace(self):
        # function spaces ----------------------------
        self.U1 = fem.VectorFunctionSpace(self.msh, ("CG", 1)) # displacement basiss
        self.D0 = fem.FunctionSpace(self.msh, ("DG", 0)) # density
        self.u, self.v = ufl.TrialFunction(self.U1), ufl.TestFunction(self.U1) # note that this is ufl
        self.u_sol = fem.Function(self.U1) 

    def assemble(self):
        def epsilon(_u):
            return ufl.sym(ufl.grad(_u))
        def sigma(_u, lmd, mu):
            return 2.0 * mu * epsilon(_u) + lmd * ufl.tr(epsilon(_u)) * ufl.Identity(len(_u))
        k = ufl.inner(sigma(self.u), ufl.grad(self.v)) * ufl.dx
        self.problem = fem.petsc.LinearProblem(k, self.f, bcs=self.bcs, petsc_options=self.petsc_options)
        self.isAssembled = True

    def applyBC(self):
        # BC
        bc_l = fem.dirichletbc(self.prob.BC_u, fem.locate_dofs_topological(self.U1, self.prob.dim-1, self.prob.BC_facetTag), self.U1)
        self.bcs = [bc_l]

        # LC 
        ds = ufl.Measure("ds", domain=self.msh, subdomain_data=self.prob.BC_facetTag) # measure for facet (surface)
        self.f = ufl.dot(self.v, self.prob.LC_force) * ds(1)


    def get_global_stiffness(self) -> np.ndarray:
        return self.problem.A
    
    def get_local_stiffness(self) -> np.ndarray:
        