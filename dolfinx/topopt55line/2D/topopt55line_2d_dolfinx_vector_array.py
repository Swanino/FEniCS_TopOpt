import ufl
import numpy as np, sklearn.metrics.pairwise as sp
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
# DOLFINX PROJECTION FUNCTION ---------------------------------
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    project_prob = fem.petsc.LinearProblem(a, l, [])
    result_sol = project_prob.solve()
    return result_sol
# SET START TIME ---------------------------------
start = time.time()
# A 55 LINE TOPOLOGY OPTIMIZATION CODE ---------------------------------
def main(nelx, nely, volfrac, penal, rmin):
    # FUNCTION DECLARATION ---------------------------------
    sigma = lambda _u: 2.0 * mu * ufl.sym(ufl.grad(_u)) + lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
    psi = lambda _u: lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))
    from pathlib import Path
    Path("output").mkdir(parents=True, exist_ok=True)
    mu, lmd = PETSc.ScalarType(0.4), PETSc.ScalarType(0.6)
    # PREPARE FINITE ELEMENT ANALYSIS ---------------------------------
    # msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right_left)
    msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.quadrilateral)
    with io.XDMFFile(MPI.COMM_WORLD, "output/density.xdmf", "w") as file:
        file.write_mesh(msh)
    U1 = fem.VectorFunctionSpace(msh, ("CG", 1))
    D0 = fem.FunctionSpace(msh, ("DG", 0))
    u, v = ufl.TrialFunction(U1), ufl.TestFunction(U1)
    u_sol, density_old, density, sensitivity = fem.Function(U1), fem.Function(D0), fem.Function(D0), fem.Function(D0)
    density.vector.array = volfrac
    # DEFINE SUPPORT ---------------------------------
    def left_clamp(x):
        return np.isclose(x[0], 0.0)
    f_dim = msh.topology.dim - 1
    bc_facets = mesh.locate_entities_boundary(msh, f_dim, left_clamp)
    u_zero = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
    bc_l = fem.dirichletbc(u_zero, fem.locate_dofs_topological(U1, f_dim, bc_facets), U1)
    bcs = [bc_l]
    # DEFINE LOAD ---------------------------------
    load_points = [(1, lambda x: np.logical_and(x[0] == nelx, x[1] <= 2))]
    facet_indices, facet_markers = [], []
    for (marker, locator) in load_points:
        facets = mesh.locate_entities(msh, f_dim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))
    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(msh, f_dim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
    f = ufl.dot(v, fem.Constant(msh, (0.0, -1.0))) * ds(1)
    # SET UP THE VARIATIONAL PROBLEM AND SOLVER ---------------------------------
    k = ufl.inner(density**penal * sigma(u), ufl.grad(v)) * ufl.dx
    problem = fem.petsc.LinearProblem(k, f, bcs=bcs)
    # PREPARE DISTANCE MATRICES FOR FILTER ---------------------------------
    t_dim = msh.topology.dim
    num_cells = msh.topology.index_map(t_dim).size_local + msh.topology.index_map(t_dim).num_ghosts
    midpoint = mesh.compute_midpoints(msh, t_dim, range(num_cells))
    distance_mat = rmin - sp.euclidean_distances(midpoint, midpoint)
    distance_mat[distance_mat < 0] = 0
    distance_sum = distance_mat.sum(1)
    # START ITERATION ---------------------------------
    loop, change = 0, 1
    x_cord, y_cord = [], []
    while change > 0.01 and loop < 2000:
        loop += 1
        density_old.vector.array = density.vector.array
        # FE-ANALYSIS ---------------------------------
        u_sol = problem.solve()
        # OBJECTIVE FUNCTION AND SENSITIVITY ---------------------------------
        objective = project_func(density**penal * psi(u_sol), D0)
        sensitivity.vector.array = -penal * (density.vector.array)**(penal-1) * project_func(psi(u_sol), D0).vector.array
        # SENSITIVITY DISTANCE FILTERING ---------------------------------
        sensitivity.vector.array = np.divide(distance_mat @ np.multiply(density.vector.array, sensitivity.vector.array), np.multiply(density.vector.array, distance_sum))
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD ---------------------------------
        l1, l2, move = 0, 1e5, 0.2
        while l2 - l1 > 1e-4:
            l_mid = 0.5 * (l2 + l1)
            if (-sensitivity.vector.array / l_mid <= 0).any(): break
            density_new = np.maximum(0.001, np.maximum(density.vector.array - move, np.minimum(1.0, np.minimum(density.vector.array + move, density.vector.array * np.sqrt(-sensitivity.vector.array/l_mid)))))
            l1, l2 = (l_mid, l2) if sum(density_new) - volfrac*num_cells > 0 else (l1, l_mid)
        density.vector.array = density_new
        # PRINT RESULTS ---------------------------------
        change = np.linalg.norm(density.vector.array - density_old.vector.array, np.inf)
        change = MPI.COMM_WORLD.bcast(change, root=0)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, sum(objective.vector.array), sum(density.vector.array)/num_cells, change))
        file.write_function(density, loop)
        # PLOT OBJECTIVE FUNCTION ---------------------------------
        x_cord.append(loop)
        y_cord.append(sum(objective.vector.array))
    file.close()
    plt.scatter(x_cord, y_cord)
    # FIND MAXIMUM VALUE IN CHART ---------------------------------
    max_id1 = np.argmax(y_cord)
    plt.plot(x_cord[max_id1], y_cord[max_id1], 'x', label=round(y_cord[max_id1], 2), ms=10)
    plt.xlabel("iteration")
    plt.ylabel("compliance")
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)
    plt.savefig('output/objective_function.jpg')
    # CHECK END TIME & SHOW RUNNING TIME ---------------------------------
    end = time.time()
    print("RUNNING Time:", end - start, "sec")
    plt.show()