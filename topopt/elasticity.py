import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt

# OPTIMIZATION PARAMETERS -------------------------------------------------
class OptimPars:
    def __init__(self, nelx=32, nely=32, nelz=32, pen=3., rmin=3., ft=1, max_iter=100, max_load=1.0, volfrac=0.5):
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.pen = pen
        self.rmin = rmin
        self.ft = ft
        self.max_iter = max_iter
        self.max_load = max_load
        self.volfrac = volfrac
    
    def __str__(self):
        return f"OptimPars(nelx={self.nelx}, nely={self.nely}, nelz={self.nelz}, pen={self.pen}, rmin={self.rmin}, ft={self.ft}, max_iter={self.max_iter}, max_load={self.max_load}, volfrac={self.volfrac})"

# LINEAR ELASTICITY PROBLEM -------------------------------------------------

