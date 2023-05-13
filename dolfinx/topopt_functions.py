import ufl
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI
# HELMHOLTZ WEAK FORM PDE FILTER FUNCTION ---------------------------------
def helm_filter(rho_n, r_min):
    V = rho_n.ufl_function_space()
    rho, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (r_min**2) * ufl.inner(ufl.grad(rho), ufl.grad(w)) * ufl.dx + rho * w * ufl.dx
    L = rho_n * w * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, [])
    rho = problem.solve()
    return rho
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
def mpi_serial_dof(msh, msh_self_idx, dfx_func, func_space):
    # GET LOCAL RANGES AND GLOBAL SIZE OF ARRAY ---------------------------------
    imap = dfx_func.function_space.dofmap.index_map
    local_range = np.asarray(imap.local_range, dtype=np.int32) * dfx_func.function_space.dofmap.index_map_bs
    size_global = imap.size_global * dfx_func.function_space.dofmap.index_map_bs
    # GATHERED RANGES & INDEX ---------------------------------
    gather_ranges = MPI.COMM_WORLD.gather(local_range, root=1)
    gather_cell_idx = MPI.COMM_WORLD.gather(msh.topology.original_cell_index[:imap.size_local], root=1)
    # COMMUNICATE LOCAL DOF COORDINATES ---------------------------------
    x = func_space.tabulate_dof_coordinates()[:imap.size_local]
    # CREATE LOCAL DOF INDEXING ARRAY ---------------------------------
    local_dof_size = MPI.COMM_WORLD.gather(len(x), root=1)
    if MPI.COMM_WORLD.rank == 1:
        # CREATE GATHERED INDEX ARRAY ---------------------------------
        global_idx = [imap.size_global * dfx_func.function_space.dofmap.index_map_bs]
        for r, x_ in zip(gather_ranges, gather_cell_idx):
            global_idx[r[0]:r[1]] = x_
        # CREATE SERIAL TO GLOBAL ---------------------------------
        ck_idx = {val: idx for idx, val in enumerate(global_idx)} # set check element index using dict comprehension
        serial_to_global = [ck_idx.get(n) for n in msh_self_idx]
    else:
        serial_to_global = None
    return size_global, gather_ranges, local_dof_size, serial_to_global
# MPI GATHER FUNCTION ---------------------------------
def mpi_gather(dfx_func, size_global, gather_ranges, serial_to_global):
    # GATHERED RANGES AND LOCAL DATA ---------------------------------
    gather_data = MPI.COMM_WORLD.gather(dfx_func.vector.array, root=1)
    # DECLARE GATHERED PARALLEL ARRAYS ---------------------------------
    global_array = np.zeros(size_global)
    dfx_func_from_global, sort_global = np.zeros(global_array.shape), np.zeros(global_array.shape)
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