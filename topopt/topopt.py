import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from util import * 
from elasticity import *
from util import *
from solver import *

# OPTIMIZATION PARAMETERS -------------------------------------------------
class OptimPars:
    def __init__(self, pen=3., rmin=3., ft=1, max_iter=100, volfrac=0.5)->None:
        self.pen = pen
        self.rmin = rmin
        self.ft = ft
        self.max_iter = max_iter
        self.volfrac = volfrac
    
    def __str__(self) -> str:
        return f"OptimPars(pen={self.pen}, rmin={self.rmin}, ft={self.ft}, max_iter={self.max_iter}, volfrac={self.volfrac})"


# main class for topology optimization
class TopOpt:
    def __init__(self, fea:ElasticPars, opt:OptimPars) -> None:
        # elasticity
        self.elasticity = Elasticity(fea)
        self.msh = self.elasticity.msh
        
        # filter radius
        rmin = np.divide(np.divide(opt.rmin, 2), np.sqrt(3))

        # initial density
        C1 = fem.FunctionSpace(self.msh, ("CG", 1)) # nodal design varia
        D0 = fem.FunctionSpace(self.msh, ("DG", 0)) # element-wise constant function space
        den_node, den_sens = fem.Function(C1), fem.Function(C1)
        density_old, density = fem.Function(D0), fem.Function(D0)
        density.x.array[:] = opt.volfrac # initial density
    

    def _solve(self, sens:np.ndarray, density:np.ndarray, type:str="oc")->np.ndarray:
        if type == "oc":
            density_new = solver_oc(self.msh, self.opt.volfrac, sens, density)
        else:
            print(f"Unknown solver type: {type}")
            exit()

        return density_new


        

