import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt

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

# LINEAR ELASTICITY PROBLEM -------------------------------------------------
class ElasticPars:
    def __init__(self, nelx=32, nely=32, nelz=32, E=1.0, mu=1.) -> None:
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.E = E
        self.mu = mu

    def __str__(self) -> str:
        return f"ElasticPars(nelx={self.nelx}, nely={self.nely}, nelz={self.nelz}, E={self.E}, mu={self.mu})"