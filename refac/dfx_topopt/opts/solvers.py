# PYTHON LIBRARIES ---------------------------------
import numpy as np
from petsc4py import PETSc

'''
    collection of petsc solvers
    ToDo:
        Create class for PETSc solver and design logic to easily handle options
'''
# SOLVER OF THE ALGEBRAIC MULTI-GRID ---------------------------------
def set_mg_solvers(self):
    # ref: https://github.com/topopt/TopOpt_in_PETSc
    '''
    special setup is required for geometric multigrid preconditioner (algibraic multigrid is slow...)
    : overrides self.problem
    '''

    # setup solver parameters
    # The fine grid solver settings
    rtol = 1.0e-5
    atol = 1.0e-50
    dtol = 1.0e5
    restart = 100
    maxitsGlobal = 200

    # Coarsegrid solver
    coarse_rtol = 1.0e-8
    coarse_atol = 1.0e-50
    coarse_dtol = 1e5
    coarse_maxits = 30
    coarse_restart = 30

    '''Number of smoothening iterations per up/down smooth_sweeps'''
    smooth_sweeps = 4
    mg_solver = PETSc.KSP().create(self.msh.comm)
    mg_solver.setOperators(self.problem.A)
    mg_solver.setType("pgmres")

    pass

'''
    collection of optimization solvers
    currently supports only OC solver (see oc_method)
'''
# SOLVER OF THE OPTIMALITY CRITERIA METHOD ---------------------------------
'''
    optimality criteria solver
    inputs:
        density(dolfinx.fem.Function): density field (DG0)
    output:
        density_new(np.ndarray): determined optimised density value of oc method
'''
def oc_method(l1, l2, move, min_val, vf, num_cells, dens: np.ndarray, sens: np.ndarray, dv: np.ndarray) -> np.ndarray:
    # USUALLY USE: l1, l2, move, min_val = 0, 1e5, 0.2, 1e-3
    while l2 - l1 > 1e-4:
        l_mid = 0.5 * (l2 + l1)
        if (-sens / l_mid <= 0).any(): break
        density_new = np.maximum(min_val, np.maximum(dens - move, np.minimum(1.0, np.minimum(dens + move, dens * np.sqrt(-sens/dv/l_mid)))))
        l1, l2 = (l_mid, l2) if sum(density_new) - vf*num_cells > 0 else (l1, l_mid)
    return density_new