import line_profiler
import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from MMA import mmasub
from matplotlib import pyplot as plt
# from memory_profiler import profile
# HELMHOLTZ WEAK FORM PDE FILTER FUNCTION ---------------------------------
def helm_filter(rho_n, r_min):
    V = rho_n.ufl_function_space()
    rho, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (r_min**2) * ufl.inner(ufl.grad(rho), ufl.grad(w)) * ufl.dx + rho * w * ufl.dx
    L = rho_n * w * ufl.dx
    # problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "gmres"})
    problem = fem.petsc.LinearProblem(a, L, [])
    rho = problem.solve()
    return rho
# DOLFINX PROJECTION FUNCTION ---------------------------------
def project_func(dfx_func, func_space):
    trial_func = ufl.TrialFunction(func_space)
    test_func = ufl.TestFunction(func_space)
    a = trial_func * test_func * ufl.dx
    l = dfx_func * test_func * ufl.dx
    # project_prob = fem.petsc.LinearProblem(a, l, petsc_options={"ksp_type": "gmres"})
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
    gather_ranges = MPI.COMM_WORLD.gather(local_range, root=1)
    # COMMUNICATE LOCAL DOF COORDINATES ---------------------------------
    x = func_space.tabulate_dof_coordinates()[:imap.size_local]
    x_glob = MPI.COMM_WORLD.gather(x, root=1)
    # CREATE LOCAL DOF INDEXING ARRAY ---------------------------------
    local_dof_size = MPI.COMM_WORLD.gather(len(x), root=1)
    serial_to_global = []
    # GATHER ---------------------------------
    if MPI.COMM_WORLD.rank == 1:
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
    gather_data = MPI.COMM_WORLD.gather(dfx_func.vector.array, root=1)
    # DECLARE GATHERED PARALLEL ARRAYS ---------------------------------
    global_array = np.zeros(size_global)
    dfx_func_from_global = np.zeros(global_array.shape)
    # GATHER ---------------------------------
    if MPI.COMM_WORLD.rank == 1:
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
    if MPI.COMM_WORLD.rank == 1:
        # PREPARE SCATTER (DELETE END OF LOCAL DOF OF SIZE) ---------------------------------
        scatter_size = np.delete(local_dof_size, len(local_dof_size) - 1)
        # BACK TO GLOBAL ARRAY ---------------------------------
        for serial, glob in enumerate(serial_to_global):
            global_from_dfx_func[glob] = dfx_func_from_global[serial]
        split_global_array = np.array_split(global_from_dfx_func, np.cumsum(scatter_size))
        raveled = [np.ravel(arr_2) for arr_2 in split_global_array]
    scatter_global_array = MPI.COMM_WORLD.scatter(raveled, root=1)
    return scatter_global_array
# SET START TIME ---------------------------------
start = time.time()
# IF WANT TO SEE LINE BY LINE PROFILING ---------------------------------
# (ACTIVE @profile, RUN THE COMMAND "mpirun -n 4 kernprof -v -l ~~.py" and "python3 -m line_profiler ~~.py.lprof > results.txt" in cmd)
# @profile
# A 55 LINE TOPOLOGY OPTIMIZATION CODE ---------------------------------
def main(nelx, nely, volfrac, penal, rmin, ft, opt_solv):
    if MPI.COMM_WORLD.rank == 0:
        print("Minimize compliance optimisation algorithm using HF")
        print("sizes: {0} X {1}, parallel core number is {2}".format(nelx, nely, MPI.COMM_WORLD.size))
        print("volume fraction: {0}, penalization power: {1:.3f}, min. filter radius: {2:.3f}".format(volfrac, penal, rmin))
        print(["Sensitivity based", "Density based"][ft] + " filtering")
        print(["OC method", "MMA"][opt_solv] + " optimizer solver\n")
    rmin = np.divide(np.divide(rmin, 2), np.sqrt(3))
    # FUNCTION DECLARATION ---------------------------------
    sigma = lambda _u: 2.0 * _mu * ufl.sym(ufl.grad(_u)) + lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
    psi = lambda _u: lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + _mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))
    from pathlib import Path
    Path(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + "_output").mkdir(parents=True, exist_ok=True)
    _mu, lmd = PETSc.ScalarType(0.4), PETSc.ScalarType(0.6)
    # PREPARE FINITE ELEMENT ANALYSIS ---------------------------------
    msh = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0.0, 0.0], [nelx, nely]]), [nelx, nely], cell_type=mesh.CellType.quadrilateral)
    t_dim = msh.topology.dim
    f_dim = msh.topology.dim - 1
    num_cells = msh.topology.index_map(t_dim).size_local
    # MESH ON CONDUCTOR PROCESS ---------------------------------
    if MPI.COMM_WORLD.size > 1:
        if MPI.COMM_WORLD.rank == 1:
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
        x0_con = MPI.COMM_WORLD.bcast(x0_con, root=1)
    U1 = fem.VectorFunctionSpace(msh, ("CG", 1))
    C1 = fem.FunctionSpace(msh, ("CG", 1))
    D0 = fem.FunctionSpace(msh, ("DG", 0))
    u, v = ufl.TrialFunction(U1), ufl.TestFunction(U1)
    u_sol, dv = fem.Function(U1), fem.Function(D0)
    density_old, density, sensitivity, den_new = fem.Function(D0), fem.Function(D0), fem.Function(D0), fem.Function(D0)
    den_node, sens_node, den_sens = fem.Function(C1), fem.Function(C1), fem.Function(C1)
    if MPI.COMM_WORLD.size == 1:
        # DON'T NEED TO DECLARE CONDUCTOR MESH ---------------------------------
        nrs = density.vector.array.size
        with io.XDMFFile(MPI.COMM_WORLD, f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + "_output/1_density_plot.xdmf", "w") as file:
            file.write_mesh(msh)
    if MPI.COMM_WORLD.size > 1:
        size_global, gather_ranges, local_dof_size, serial_to_global = mpi_serial_dof(density, D0, x0_con)
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
    problem = fem.petsc.LinearProblem(k, f, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    # INITIALIZE MMA OPTIMIZER ---------------------------------
    if opt_solv == 1:
        if MPI.COMM_WORLD.size == 1:
            m = 1  # THE NUMBER OF GENERAL CONSTRAINTS
            xmin = np.zeros((nrs, 1))                       # COLUMN VECTOR WITH THE LOWER BOUNDS FOR THE VARIABLES x_j
            xmax = np.ones((nrs, 1))                        # COLUMN VECTOR WITH THE UPPER BOUNDS FOR THE VARIABLES x_j
            xold1 = density.vector.array[np.newaxis].T      # xval, ONE ITERATION AGO (PROVIDED THAT ITER>1)
            xold2 = density.vector.array[np.newaxis].T      # xval, TWO ITERATIONS AGO (PROVIDED THAT ITER>2)
            low = np.ones((nrs, 1))                         # COLUMN VECTOR WITH THE LOWER ASYMPTOTES FROM THE PREVIOUS ITERATION (PROVIDED THAT ITER>1)
            upp = np.ones((nrs, 1))                         # COLUMN VECTOR WITH THE UPPER ASYMPTOTES FROM THE PREVIOUS ITERATION (PROVIDED THAT ITER>1)
            a0 = 1.0                                        # THE CONSTANTS a_0 IN THE TERM a_0*z
            a = np.zeros((m, 1))                            # COLUMN VECTOR WITH THE CONSTANTS a_i IN THE TERMS a_i*z
            c = 1e4 * np.ones((m, 1))                       # COLUMN VECTOR WITH THE CONSTANTS c_i IN THE TERMS c_i*y_i
            d = np.zeros((m, 1))                            # COLUMN VECTOR WITH THE CONSTANTS d_i IN THE TERMS 0.5*d_i*(y_i)^2
            move_mma = 0.2
        if MPI.COMM_WORLD.size > 1:
            if MPI.COMM_WORLD.rank == 1:
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
    x_cord, y_cord, x_cord1, y_cord1, x_cord2, y_cord2 = [], [], [], [], [], []
    iter_data_txt = open(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/0_iteration_data_{MPI.COMM_WORLD.rank}.txt", 'a')
    flag = 0
    if opt_solv == 0: flag = 1e-2
    elif opt_solv == 1: flag = 1e-3
    while change > flag and loop < 2000:
        loop += 1
        density_old.vector.array = density.vector.array
        # FE-ANALYSIS ---------------------------------
        u_sol = problem.solve()
        # OBJECTIVE FUNCTION ---------------------------------
        objective = project_func(density**penal * psi(u_sol), D0)
        # SENSITIVITY NODAL PROJECTION (DG0 TO CG1) ---------------------------------
        den_node = project_func(density, C1)
        psi_node = project_func(psi(u_sol), C1)
        sens_node.vector.array = -penal * (den_node.vector.array) ** (penal - 1) * psi_node.vector.array
        # SENSITIVITY DISTANCE FILTERING ---------------------------------
        if ft == 0:
            # PREPARE DENSITY HELMHOLTZ FILTERING ---------------------------------
            den_sens.vector.array = np.multiply(den_node.vector.array, sens_node.vector.array)
            # HELMHOLTZ FILTERING ---------------------------------
            den_sens_til_node = helm_filter(den_sens, rmin)
            # FILTERED HELMHOLTZ VARIABLE PROJECTION (CG1 TO DG0) ---------------------------------
            density_til = project_func(den_sens_til_node, D0)
            # SENSITIVITY ANALYSIS ---------------------------------
            sensitivity.vector.array = np.divide(density_til.vector.array, np.maximum(1e-3, density.vector.array))
        # DENSITY DISTANCE FILTERING ---------------------------------
        elif ft == 1:
            sensitivity = project_func(helm_filter(sens_node, rmin), D0)
            dv = project_func(helm_filter(project_func(dv, C1), rmin), D0)
        # GATHERING VARIABLES ---------------------------------
        if MPI.COMM_WORLD.size > 1:
            de_gather, de_glo = mpi_gather(density, size_global, gather_ranges, serial_to_global)
            sensitivity_gather, sens_glo = mpi_gather(sensitivity, size_global, gather_ranges, serial_to_global)
            dv_gather, dv_glo = mpi_gather(dv, size_global, gather_ranges, serial_to_global)
            if opt_solv == 1:
                objective_gather, obj_glo = mpi_gather(objective, size_global, gather_ranges, serial_to_global)
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD ---------------------------------
        if opt_solv == 0:
            if MPI.COMM_WORLD.size == 1:
                l1, l2, move_oc = 0, 1e5, 0.2
                while l2 - l1 > 1e-4:
                    l_mid = 0.5 * (l2 + l1)
                    if (-sensitivity.vector.array / l_mid <= 0).any(): break
                    density_new = np.maximum(0.001, np.maximum(density.vector.array - move_oc, np.minimum(1.0, np.minimum(density.vector.array + move_oc, density.vector.array * np.sqrt(-sensitivity.vector.array / dv.vector.array / l_mid)))))
                    l1, l2 = (l_mid, l2) if sum(density_new) - volfrac*num_cells > 0 else (l1, l_mid)
            if MPI.COMM_WORLD.size > 1:
                if MPI.COMM_WORLD.rank == 1:
                    l1, l2, move_oc = 0, 1e5, 0.2
                    while l2 - l1 > 1e-4:
                        l_mid = 0.5 * (l2 + l1)
                        if (-sensitivity_gather / l_mid <= 0).any(): break
                        density_new_con = np.maximum(0.001, np.maximum(de_gather - move_oc, np.minimum(1.0, np.minimum(de_gather + move_oc, de_gather * np.sqrt(-sensitivity_gather/dv_gather/l_mid)))))
                        l1, l2 = (l_mid, l2) if sum(density_new_con) - volfrac*num_cells_con > 0 else (l1, l_mid)
                else:
                    density_new_con = None
                density_new_con = MPI.COMM_WORLD.bcast(density_new_con, root=1)
                density_new = mpi_gathered_scatter(density_new_con, de_glo, local_dof_size, serial_to_global)
        # DESIGN UPDATE BY THE METHOD OF MOVING ASYMPTOTES ---------------------------------
        elif opt_solv == 1:
            if MPI.COMM_WORLD.size == 1:
                # MMA INPUT PARAMETERS ---------------------------------
                mu0 = 1.0  # SCALE FACTOR FOR OBJECTIVE FUNCTION
                mu1 = 1.0  # SCALE FACTOR FOR VOLUME CONSTRAINT FUNCTION
                f0val = mu0 * sum(objective.vector.array)
                df0dx = mu0 * sensitivity.vector.array[np.newaxis].T
                fval = mu1 * np.array([[sum(density.vector.array) / nrs - volfrac]])
                dfdx = mu1 * np.divide(dv.vector.array, np.multiply(nrs, volfrac))[np.newaxis]
                xval = density.vector.array[np.newaxis].T
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(m, nrs, loop, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move_mma)
                xold2 = xold1
                xold1 = xval
                density_new = xmma.copy().flatten()
            if MPI.COMM_WORLD.size > 1:
                if MPI.COMM_WORLD.rank == 1:
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
                density_new_con = MPI.COMM_WORLD.bcast(density_new_con, root=1)
                density_new = mpi_gathered_scatter(density_new_con, de_glo, local_dof_size, serial_to_global)
        # FILTERING DESIGN VARIABLES ---------------------------------
        if ft == 0: density.vector.array = density_new
        elif ft == 1:
            den_new.vector.array = density_new
            density = project_func(helm_filter(project_func(den_new, C1), rmin), D0)
        # PRINT RESULTS ---------------------------------
        change = np.linalg.norm(density.vector.array - density_old.vector.array, np.inf)
        if MPI.COMM_WORLD.size == 1:
            print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, ch.: {4:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array) / num_cells, change))
            print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, ch.: {4:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array) / num_cells, change), file=iter_data_txt)
        if MPI.COMM_WORLD.size > 1:
            change = MPI.COMM_WORLD.bcast(change, root=1)
            num_cells_con = MPI.COMM_WORLD.bcast(num_cells_con, root=1)
            print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array)/num_cells, sum(density_new_con)/num_cells_con, change))
            print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(objective.vector.array), MPI.COMM_WORLD.rank, sum(density.vector.array)/num_cells, sum(density_new_con)/num_cells_con, change), file=iter_data_txt)
        # MPI GATHER (DENSITY NEW) ---------------------------------
        if MPI.COMM_WORLD.size == 1:
            # WRITE SINGLE CORE DENSITY FUNCTION ON PARAVIEW ---------------------------------
            file.write_function(density, loop)
        if MPI.COMM_WORLD.size > 1:
            density_gather, density_gather_glo = mpi_gather(density, size_global, gather_ranges, serial_to_global)
            if MPI.COMM_WORLD.rank == 1:
                density_con.vector.array = density_gather
                # WRITE MULTI-CORE DENSITY FUNCTION ON PARAVIEW ---------------------------------
                file.write_function(density_con, loop)
        # APPEND EACH POINT OBJECTIVE FUNCTION & SENSITIVITY ---------------------------------
        x_cord.append(loop)
        y_cord.append(sum(objective.vector.array))
        x_cord1.append(loop)
        y_cord1.append(sum(sensitivity.vector.array))
        x_cord2.append(loop)
        y_cord2.append(round(sum(density.vector.array) / num_cells, 3))
    if MPI.COMM_WORLD.size == 1:
        file.close()
    if MPI.COMM_WORLD.size > 1:
        if MPI.COMM_WORLD.rank == 1:
            file.close()
    # CLOSED ITERATION DATA TXT FILE ---------------------------------
    iter_data_txt.close()
    # PLOT AND FIND MAXIMUM VALUE IN OBJECTIVE FUNCTION CHART ---------------------------------
    plt.figure(1)
    plt.scatter(x_cord, y_cord, color='blue')
    max_id1 = np.argmax(y_cord)
    plt.plot(x_cord[max_id1], y_cord[max_id1], color='red', marker='x', label=round(y_cord[max_id1], 2), ms=10)
    plt.xlabel('iteration', labelpad=5)
    plt.ylabel('compliance', labelpad=10)
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.savefig(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/2_compliance_pros_{MPI.COMM_WORLD.rank}.jpg")
    # PLOT AND FIND MAXIMUM VALUE IN SENSITIVITY CHART ---------------------------------
    plt.figure(2)
    plt.scatter(x_cord1, y_cord1, color='orange')
    max_id2 = np.argmax(y_cord1)
    plt.plot(x_cord1[max_id2], y_cord1[max_id2], color='blue', marker='x', label=round(y_cord1[max_id2], 2), ms=10)
    plt.xlabel('iteration', labelpad=5)
    plt.ylabel('sensitivity', labelpad=-5)
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(loc='lower right', fontsize=10)
    plt.savefig(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/3_sensitivity_pros_{MPI.COMM_WORLD.rank}.jpg")
    # PLOT AND FIND MAXIMUM VALUE IN LOCAL VOLUME FRACTION CHART ---------------------------------
    plt.figure(3)
    plt.scatter(x_cord2, y_cord2, color='green')
    max_id3 = np.argmax(y_cord2)
    plt.plot(x_cord2[max_id3], y_cord2[max_id3], color='red', marker='x', label=round(y_cord2[max_id3], 2), ms=10)
    plt.xlabel('iteration', labelpad=5)
    plt.ylabel('local volume fraction', labelpad=5)
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    plt.savefig(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/4_local_volume_fraction_pros_{MPI.COMM_WORLD.rank}.jpg")
    # CHECK END TIME & SHOW RUNNING TIME ---------------------------------
    end = time.time()
    time_txt = open(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_solv] + ["_sens_ft", "_den_ft"][ft] + f"_output/0_running_time.txt", 'a')
    print(f"RUNNING Time(core {MPI.COMM_WORLD.rank}):", end - start, "sec")
    print(f"RUNNING Time(core {MPI.COMM_WORLD.rank}):", end - start, "sec", file=time_txt)
    time_txt.close()
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
    if len(sys.argv)>3: volfrac  = float(sys.argv[4])
    if len(sys.argv)>4: penal    = float(sys.argv[5])
    if len(sys.argv)>5: rmin     = float(sys.argv[6])
    if len(sys.argv)>6: ft       = int(sys.argv[7])
    if len(sys.argv)>7: opt_solv = int(sys.argv[8])
    main(nelx, nely, volfrac, penal, rmin, ft, opt_solv)