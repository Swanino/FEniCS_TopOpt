import ufl
import numpy as np  
from elasticity import *
from input_files.cantilever import Cantilever2D
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
    def assemble(self):
        pass
    @abstractmethod
    def applyBC(self):
        pass
    

# define macro
def epsilon(_u):
    return ufl.sym(ufl.grad(_u))
def sigma(_u, lmd, mu):
    return 2.0 * mu * epsilon(_u) + lmd * ufl.tr(epsilon(_u)) * ufl.Identity(len(_u))


