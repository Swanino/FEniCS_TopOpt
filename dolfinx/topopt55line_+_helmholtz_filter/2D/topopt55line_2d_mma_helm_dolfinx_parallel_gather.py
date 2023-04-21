import line_profiler
import ufl
import numpy as np, sklearn.metrics.pairwise as sp
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from MMA import mmasub
# from memory_profiler import profile
# DOLFINX PROJECTION FUNCTION ---------------------------------
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    project_prob = fem.petsc.LinearProblem(a, l, [])
    result_sol = project_prob.solve()
    return result_sol
# SERIAL DOF FUNCTION ---------------------------------
def mpi_serial_dof(dfx_func, func_space, x0_con):
    # GET LOCAL RANGES AND GLOBAL SIZE OF ARRAY ---------------------------------
    imap = dfx_func.function_space.dofmap.index_map
    local_range = np.asarray(imap.local_range, dtype=np.int32) * dfx_func.function_space.dofmap.index_map_bs
    size_global = imap.size_global * dfx_func.function_space.dofmap.index_map_bs
    # GATHERED RANGES ---------------------------------
    gather_ranges = MPI.COMM_WORLD.gather(local_range, root=0)
    # COMMUNICATE LOCAL DOF COORDINATES ---------------------------------
    x = func_space.tabulate_dof_coordinates()[:imap.size_local]
    x_glob = MPI.COMM_WORLD.gather(x, root=0)
    # CREATE LOCAL DOF INDEXING ARRAY ---------------------------------
    local_dof_size = MPI.COMM_WORLD.gather(len(x), root=0)
    serial_to_global = []
    # GATHER ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        # CREATE ARRAY WITH ALL COORDINATES ---------------------------------
        global_x = np.zeros((size_global, 3))
        for r, x_ in zip(gather_ranges, x_glob):
            global_x[r[0]:r[1], :] = x_
        for coord in x0_con:
            serial_to_global.append(np.abs(global_x - coord).sum(axis=1).argmin())
    return size_global, gather_ranges, local_dof_size, serial_to_global
# MPI GATHER FUNCTION ---------------------------------
def mpi_gather(dfx_func, size_global, gather_ranges, serial_to_global):
    # GATHERED RANGES AND LOCAL DATA ---------------------------------
    gather_data = MPI.COMM_WORLD.gather(dfx_func.vector.array, root=0)
    # DECLARE GATHERED PARALLEL ARRAYS ---------------------------------
    global_array = np.zeros(size_global)
    dfx_func_from_global = np.zeros(global_array.shape)
    # GATHER ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        # CREATE ARRAY RECEIVED ALL VALUES ---------------------------------
        for r, d in zip(gather_ranges, gather_data):
            global_array[r[0]:r[1]] = d
        # CREATE SORTED ARRAY FROM ---------------------------------
        for serial, glob in enumerate(serial_to_global):
            dfx_func_from_global[serial] = global_array[glob]
    return dfx_func_from_global, global_array
# MPI SCATTER FUNCTION ---------------------------------
def mpi_gathered_scatter(dfx_func_from_global, global_array, local_dof_size, serial_to_global):
    global_from_dfx_func, raveled = np.zeros(global_array.shape), []
    # SCATTER ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        # PREPARE SCATTER (DELETE END OF LOCAL DOF OF SIZE) ---------------------------------
        scatter_size = np.delete(local_dof_size, len(local_dof_size) - 1)
        # BACK TO GLOBAL ARRAY ---------------------------------
        for serial, glob in enumerate(serial_to_global):
            global_from_dfx_func[glob] = dfx_func_from_global[serial]
        split_global_array = np.array_split(global_from_dfx_func, np.cumsum(scatter_size))
        raveled = [np.ravel(arr_2) for arr_2 in split_global_array]
    scatter_global_array = MPI.COMM_WORLD.scatter(raveled, root=0)
    return scatter_global_array
# SET START TIME ---------------------------------
start = time.time()
# IF WANT TO SEE LINE BY LINE PROFILING ---------------------------------
# (ACTIVE @profile, RUN THE COMMAND "mpirun -n 4 kernprof -v -l ~~.py" and "python3 -m line_profiler ~~.py.lprof > results.txt" in cmd)
# @profile
# A 55 LINE TOPOLOGY OPTIMIZATION CODE ---------------------------------
def main(nelx, nely, volfrac, penal, rmin, ft, opt_solv):
    if MPI.COMM_WORLD.rank == 0:
        print("Minimize compliance optimisation algorithm using EDF")
        print("sizes: {0} X {1}, parallel core number is {2}".format(nelx, nely, MPI.COMM_WORLD.size))
        print("volume fraction: {0}, penalization power: {1:.3f}, min. filter radius: {2:.3f}".format(volfrac, penal, rmin))
        print(["Sensitivity based", "Density based"][ft] + " filtering")
        print(["OC method", "MMA"][opt_solv] + " optimizer solver\n")
    # FUNCTION DECLARATION ---------------------------------
    sigma = lambda _u: 2.0 * _mu * ufl.sym(ufl.grad(_u)) + lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
    psi = lambda _u: lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + _mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))
    from pathlib import Path
    Path(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + "_output").mkdir(parents=True, exist_ok=True)
    _mu, lmd = PETSc.ScalarType(0.4), PETSc.ScalarType(0.6)
    # PREPARE FINITE ELEMENT ANALYSIS ---------------------------------
    # msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right_left)
    msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.quadrilateral)
    t_dim = msh.topology.dim
    f_dim = msh.topology.dim - 1
    # MESH ON CONDUCTOR PROCESS ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        # msh_con = mesh.create_rectangle(MPI.COMM_SELF, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.right_left)
        msh_con = mesh.create_rectangle(MPI.COMM_SELF, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.quadrilateral)
        num_cells_con = msh_con.topology.index_map(msh_con.topology.dim).size_local
        D0_con = fem.FunctionSpace(msh_con, ("DG", 0))
        x0_con = D0_con.tabulate_dof_coordinates()
        density_con = fem.Function(D0_con)
        nrs = density_con.vector.array.size
        # PLOT CONDUCTOR MESH DOMAIN ---------------------------------
        with io.XDMFFile(MPI.COMM_SELF, f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + "_output/1_density_plot.xdmf", "w") as file:
            file.write_mesh(msh_con)
    else:
        num_cells_con = None
        x0_con = None
        density_con = None
        nrs = None
    x0_con = MPI.COMM_WORLD.bcast(x0_con, root=0)
    U1 = fem.VectorFunctionSpace(msh, ("CG", 1))
    D0 = fem.FunctionSpace(msh, ("DG", 0))
    u, v = ufl.TrialFunction(U1), ufl.TestFunction(U1)
    u_sol, dv = fem.Function(U1), fem.Function(D0)
    density_old, density = fem.Function(D0), fem.Function(D0)
    size_global, gather_ranges, local_dof_size, serial_to_global = mpi_serial_dof(density, D0, x0_con)
    # nrs = round((nely*nelx) / MPI.COMM_WORLD.size)
    # print(f"check {density.vector.array.size}\n")
    density.vector.array = volfrac
    dv.vector.array = 1.0
    # DEFINE SUPPORT ---------------------------------
    def left_clamp(x):
        return np.isclose(x[0], 0.0)
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
    k = ufl.inner(density ** penal * sigma(u), ufl.grad(v)) * ufl.dx
    # problem = fem.petsc.LinearProblem(k, f, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 200, "mat_mumps_icntl_24": 1,})
    problem = fem.petsc.LinearProblem(k, f, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    # PREPARE DISTANCE MATRICES FOR FILTER ---------------------------------
    num_cells = msh.topology.index_map(t_dim).size_local
    midpoint = mesh.compute_midpoints(msh, t_dim, range(num_cells))
    distance_mat = rmin - sp.euclidean_distances(midpoint, midpoint)
    distance_mat[distance_mat < 0] = 0
    distance_sum = distance_mat.sum(1)
    # INITIALIZE MMA OPTIMIZER ---------------------------------
    if opt_solv == 1:
        if MPI.COMM_WORLD.rank == 0:
            m = 1                                           # THE NUMBER OF GENERAL CONSTRAINTS
            xmin = np.zeros((nrs, 1))                       # COLUMN VECTOR WITH THE LOWER BOUNDS FOR THE VARIABLES x_j
            xmax = np.ones((nrs, 1))                        # COLUMN VECTOR WITH THE UPPER BOUNDS FOR THE VARIABLES x_j
            xold1 = density_con.vector.array[np.newaxis].T  # xval, ONE ITERATION AGO (PROVIDED THAT ITER>1)
            xold2 = density_con.vector.array[np.newaxis].T  # xval, TWO ITERATIONS AGO (PROVIDED THAT ITER>2)
            low = np.ones((nrs, 1))                         # COLUMN VECTOR WITH THE LOWER ASYMPTOTES FROM THE PREVIOUS ITERATION (PROVIDED THAT ITER>1)
            upp = np.ones((nrs, 1))                         # COLUMN VECTOR WITH THE UPPER ASYMPTOTES FROM THE PREVIOUS ITERATION (PROVIDED THAT ITER>1)
            a0 = 1.0                                        # THE CONSTANTS a_0 IN THE TERM a_0*z
            a = np.zeros((m, 1))                            # COLUMN VECTOR WITH THE CONSTANTS a_i IN THE TERMS a_i*z
            c = 1e4 * np.ones((m, 1))                       # COLUMN VECTOR WITH THE CONSTANTS c_i IN THE TERMS c_i*y_i
            d = np.zeros((m, 1))                            # COLUMN VECTOR WITH THE CONSTANTS d_i IN THE TERMS 0.5*d_i*(y_i)^2
            move_mma = 0.2
    # START ITERATION ---------------------------------
    loop, change = 0, 1
    iter_data_txt = open(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/0_iteration_data.txt", 'a')
    flag = 0
    if opt_solv == 0: flag = 1e-2
    elif opt_solv == 1: flag = 1e-3
    while change > flag and loop < 2000:
        loop += 1
        density_old.vector.array = density.vector.array
        # FE-ANALYSIS ---------------------------------
        u_sol = problem.solve()
        # OBJECTIVE FUNCTION AND SENSITIVITY ---------------------------------
        objective = project_func(density**penal * psi(u_sol), D0)
        sensitivity = project_func(-penal * density**(penal-1) * psi(u_sol), D0)
        # sen = mpi_gather(project_func(-penal * density**(penal-1) * psi(u_sol), D0), size_global, gather_ranges, serial_to_global)
        # SENSITIVITY DISTANCE FILTERING ---------------------------------
        if ft == 0:
            sensitivity.vector.array = np.divide(distance_mat @ np.multiply(density.vector.array, sensitivity.vector.array), np.multiply(np.maximum(0.003, density.vector.array), distance_sum))
        # DENSITY DISTANCE FILTERING ---------------------------------
        elif ft == 1:
            sensitivity.vector.array = distance_mat @ np.divide(sensitivity.vector.array, distance_sum)
            dv.vector.array = distance_mat @ np.divide(dv.vector.array, distance_sum)
        # GATHERING VARIABLES ---------------------------------
        de_gather, de_glo = mpi_gather(density, size_global, gather_ranges, serial_to_global)
        sensitivity_gather, sens_glo = mpi_gather(sensitivity, size_global, gather_ranges, serial_to_global)
        dv_gather, dv_glo = mpi_gather(dv, size_global, gather_ranges, serial_to_global)
        if opt_solv == 1:
            objective_gather, obj_glo = mpi_gather(objective, size_global, gather_ranges, serial_to_global)
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD ---------------------------------
        if opt_solv == 0:
            if MPI.COMM_WORLD.rank == 0:
                l1, l2, move_oc = 0, 1e5, 0.2
                while l2 - l1 > 1e-4:
                    l_mid = 0.5 * (l2 + l1)
                    if (-sensitivity_gather / l_mid <= 0).any(): break
                    density_new_con = np.maximum(0.001, np.maximum(de_gather - move_oc, np.minimum(1.0, np.minimum(de_gather + move_oc, de_gather * np.sqrt(-sensitivity_gather/dv_gather/l_mid)))))
                    l1, l2 = (l_mid, l2) if sum(density_new_con) - volfrac*num_cells_con > 0 else (l1, l_mid)
            else:
                density_new_con = None
            density_new_con = MPI.COMM_WORLD.bcast(density_new_con, root=0)
            density_new = mpi_gathered_scatter(density_new_con, de_glo, local_dof_size, serial_to_global)
        # DESIGN UPDATE BY THE METHOD OF MOVING ASYMPTOTES ---------------------------------
        elif opt_solv == 1:
            if MPI.COMM_WORLD.rank == 0:
                # MMA INPUT PARAMETERS ---------------------------------
                mu0 = 1.0   # SCALE FACTOR FOR OBJECTIVE FUNCTION
                mu1 = 1.0   # SCALE FACTOR FOR VOLUME CONSTRAINT FUNCTION
                f0val = mu0 * sum(objective_gather)
                df0dx = mu0 * sensitivity_gather[np.newaxis].T
                fval = mu1 * np.array([[sum(de_gather)/nrs-volfrac]])
                dfdx = mu1 * np.divide(dv_gather, np.multiply(nrs, volfrac))[np.newaxis]
                xval = de_gather[np.newaxis].T
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(m, nrs, loop, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move_mma)
                xold2 = xold1
                xold1 = xval
                density_new_con = xmma.copy().flatten()
            else:
                density_new_con = None
            density_new_con = MPI.COMM_WORLD.bcast(density_new_con, root=0)
            density_new = mpi_gathered_scatter(density_new_con, de_glo, local_dof_size, serial_to_global)
        # FILTERING DESIGN VARIABLES ---------------------------------
        if ft == 0: density.vector.array = density_new
        elif ft == 1: density.vector.array = np.divide(distance_mat @ density_new, distance_sum)
        # PRINT RESULTS ---------------------------------
        change = np.linalg.norm(density.vector.array - density_old.vector.array, np.inf)
        change = MPI.COMM_WORLD.bcast(change, root=0)
        num_cells_con = MPI.COMM_WORLD.bcast(num_cells_con, root=0)
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array)/num_cells, sum(density_new_con)/num_cells_con, change))
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array)/num_cells, sum(density_new_con)/num_cells_con, change), file=iter_data_txt)
        # MPI GATHER (DENSITY NEW) ---------------------------------
        # density_gather, density_gather_glo, density_gather_stg, density_gather_sct_size = mpi_gather(density, D0, x0_con)
        density_gather, density_gather_glo = mpi_gather(density, size_global, gather_ranges, serial_to_global)
        # oj = mpi_gather(nelx, nely, objective, D0)
        if MPI.COMM_WORLD.rank == 0:
            density_con.vector.array = density_gather
            # WRITE DENSITY FUNCTION ON PARAVIEW ---------------------------------
            file.write_function(density_con, loop)
    if MPI.COMM_WORLD.rank == 0:
        file.close()
    # CLOSED ITERATION DATA TXT FILE ---------------------------------
    iter_data_txt.close()
    # CHECK END TIME & SHOW RUNNING TIME ---------------------------------
    end = time.time()
    time_txt = open(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/0_running_time.txt", 'a')
    print(f"RUNNING Time(core {MPI.COMM_WORLD.rank}):", end - start, "sec")
    print(f"RUNNING Time(core {MPI.COMM_WORLD.rank}):", end - start, "sec", file=time_txt)
    time_txt.close()
    # plt.show()
# CALL MAIN FUNCTION ---------------------------------
if __name__ == "__main__":
    # DEFAULT INPUT PARAMETERS ---------------------------------
    nelx = 60
    nely = 20
    volfrac = 0.5
    penal = 3.0
    rmin = 3.0
    # ft == 0 -> sensitivity, ft == 1 -> density
    ft = 0
    # opt_solv == 0 -> OC, opt_solv == 1 -> MMA
    opt_solv = 0
    import sys
    if len(sys.argv)>1: nelx     = int(sys.argv[1])
    if len(sys.argv)>2: nely     = int(sys.argv[2])
    if len(sys.argv)>3: volfrac  = float(sys.argv[3])
    if len(sys.argv)>4: penal    = float(sys.argv[4])
    if len(sys.argv)>5: rmin     = float(sys.argv[5])
    if len(sys.argv)>6: ft       = int(sys.argv[6])
    if len(sys.argv)>7: opt_solv = int(sys.argv[7])
    main(nelx, nely, volfrac, penal, rmin, ft, opt_solv)