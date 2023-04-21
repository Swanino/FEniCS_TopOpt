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
    def __init__(self, msh:mesh , opt:OptimPars) -> None:
        # filter radius
        rmin = np.divide(np.divide(opt.rmin, 2), np.sqrt(3))

        # initial density


