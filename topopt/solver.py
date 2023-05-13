''' 
    collection of optimization solvers
    currently supports only OC solver (see solver_oc): this is parallel in that it uses   )
'''

'''
    optimality criteria solver
    inputs
        density: density field (DG0 function space)

'''
from dolfinx import fem, mesh, io
import numpy as np
from petsc4py import PETSc


def solver_oc(msh: mesh.Mesh, volfrac:np.float64, sens:np.ndarray, density:np.ndarray) -> np.ndarray:
    l1, l2, move = 0, 1e5, 0.2
    dens_min = 0.001
    t_dim = msh.topology.dim # dim of problem
    num_cells = msh.topology.index_map(t_dim).size_local # num of element in local process
    while l2 - l1 > 1e-4:
        l_mid = 0.5 * (l2 + l1)
        if (-sens/l_mid <= 0).any(): break 
        density_new = np.maximum(dens_min, np.maximum(density - move, np.minimum(1.0, np.minimum(density + move, density * np.sqrt(-sens/l_mid)))))
        l1, l2 = (l_mid, l2) if sum(density_new) - volfrac*num_cells > 0 else (l1, l_mid)
    return density_new

