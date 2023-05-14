''' 
this defines problem 
'''
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
from topopt import *

'''
virtual class for topology optimization problem
'''
class problem:
    def __init__(self, elpars:ElasticPars, optpars:OptimPars) -> None:
        '''
        implement here
        '''
        self.elpars = elpars
        self.optpars = optpars

    def objective(self, u_sol:fem.Function)->float:
        '''
        implement here
        '''
        return 0.0
    
    def obj_sensitivity(self, u_sol:fem.Function):
        '''
        implement here
        '''
        return 0.0
    def constraint(self, u_sol:fem.Function)->float:
        '''
        implement here
        '''
        return 0.0
    def cons_sensitivity(self, u_sol:fem.Function):
        '''
        implement here
        '''
        return 0.0

class compliance:
    def __init__(self) -> None:
        self.sigma = lambda _u: 2.0 * self.elpars.mu * ufl.sym(ufl.grad(_u)) + self.elpars.lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
        self.psi = lambda _u: self.elpars.lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + self.elpars.mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))

    def 
        # OBJECTIVE FUNCTION ---------------------------------
        objective = project_func(density**penal * psi(u_sol), D0)
        # SENSITIVITY NODAL PROJECTION (DG0 TO CG1) ---------------------------------
        den_node = project_func(density, C1)
        psi_node = project_func(psi(u_sol), C1)
        sens_node = -penal * (den_node.vector.array)**(penal-1) * psi_node.vector.array

