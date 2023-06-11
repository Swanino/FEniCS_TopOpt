# PYTHON LIBRARIES ---------------------------------
import numpy as np
from pathlib import Path
from dolfinx import io
from mpi4py import MPI

# PACKAGE LIBRARIES ---------------------------------
from ..fem.params import OptimParams
from ..fem.mesh import MeshGen

'''
    make save directory folder
    input:
        opt_params: OptimParams
    
    output:
        make directory folder
'''
def make_folder(opt_params: OptimParams):
    if MPI.COMM_WORLD.rank == 0:
        Path(f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_params.opt_solv] + ["_sens_ft", "_den_ft"][opt_params.ft] + "_output").mkdir(parents=True, exist_ok=True)

'''
    plot mesh and conductor mesh domain to xdmf file
    inputs:
        msh (dolfinx.mesh): mesh domain which fits user's needs
        msh_con (dolfinx.mesh): conductor mesh domain which fits user's needs
    
    output:
        filename: xdmf file
'''
def plot_mesh(msh_info: MeshGen, opt_params: OptimParams) -> str:
    if MPI.COMM_WORLD.size == 1:
        file = io.XDMFFile(MPI.COMM_WORLD, f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_params.opt_solv] + ["_sens_ft", "_den_ft"][opt_params.ft] + "_output/1_density_plot.xdmf", "w")
        file.write_mesh(msh_info.msh)
    elif MPI.COMM_WORLD.size > 1:
        if MPI.COMM_WORLD.rank == 1:
            file = io.XDMFFile(MPI.COMM_SELF, f"{MPI.COMM_WORLD.size}_core" + ["_OC", "_MMA"][opt_params.opt_solv] + ["_sens_ft", "_den_ft"][opt_params.ft] + "_output/1_density_plot.xdmf", "w")
            file.write_mesh(msh_info.msh_con)
        else:
            file = None
    return file

'''
    print results on terminal windows
    inputs:
        iter_data_txt: a command to create a txt file in the desired directory folder using the 'open' command
        msh_info: MeshGen
        loop(int): flag of optimisation iteration process
        obj(np.ndarray): vector array of objective(compliance) function
        den_old(np.ndarray): vector array of density_old function
        den_opt(np.ndarray): vector array of optimised density function at each iteration
        den_new_con(np.ndarray): same as 'density_new' array (determined in gathered conductor process)
'''
# PRINT RESULTS ON TERMINAL WINDOWS ---------------------------------
def print_results(iter_data_txt, msh_info: MeshGen, loop, obj: np.ndarray, den_old: np.ndarray, den_opt: np.ndarray, den_new_con: np.ndarray):
    change = np.linalg.norm(den_opt - den_old, np.inf)
    if MPI.COMM_WORLD.size == 1:
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, ch.: {4:.3f}".format(loop, sum(obj), MPI.COMM_WORLD.rank, sum(den_opt) / msh_info.num_cells, change))
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, ch.: {4:.3f}".format(loop, sum(obj), MPI.COMM_WORLD.rank, sum(den_opt) / msh_info.num_cells, change), file=iter_data_txt)
    elif MPI.COMM_WORLD.size > 1:
        change = MPI.COMM_WORLD.bcast(change, root=1)
        cells_con = MPI.COMM_WORLD.bcast(msh_info.num_cells_con, root=1)
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(obj), MPI.COMM_WORLD.rank, sum(den_opt) / msh_info.num_cells, sum(den_new_con) / cells_con, change))
        print("it.: {0}, obj.: {1:.3f}, core {2} vol.: {3:.3f}, total vol.: {4:.3f}, ch.: {5:.3f}".format(loop, sum(obj), MPI.COMM_WORLD.rank, sum(den_opt) / msh_info.num_cells, sum(den_new_con) / cells_con, change), file=iter_data_txt)

'''
    ---------- notice ----------
    The functions below are constructed so that they can be used appropriately in parallel computation.
    Note that errors may occur if used when calculating with a single core (non-mpi computation).
'''
# SERIAL DOF FUNCTION ---------------------------------
'''
    This function maps the indexing relationship between the mesh declared in parallel computation and single core computation.
    inputs:
        msh(dolfinx.mesh): mesh declared in parallel
        msh_self_idx(dolfinx.mesh): the topological original cell index of single core
        dfx_func(dolfinx.fem.Function): paralleled function formed from the D0 function space
        (Code that satisfies the function space with other dimensions needs to be updated later)
        func_space(dolfinx.fem.Function): paralleled D0 function space
    
    outputs:
        size_global: size information of global array
        gather_ranges(numpy → list): gather the local index range assigned to each core
        local_dof_size(numpy → list): gather local data
        serial_to_global(list): indexing information between mesh indexing declared in parallel and single core
'''
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
        ck_idx = {val: idx for idx, val in enumerate(global_idx)} # using dict comprehension
        serial_to_global = [ck_idx.get(n) for n in msh_self_idx]
    else:
        serial_to_global = None
    return size_global, gather_ranges, local_dof_size, serial_to_global

# MPI GATHER FUNCTION ---------------------------------
'''
    inputs:
        dfx_func(dolfinx.fem.Function): paralleled function formed from the D0 function space
        (Code that satisfies the function space with other dimensions needs to be updated later)
        size_global: size information of global array
        gather_ranges(numpy → list): gather the local index range assigned to each core
        serial_to_global(list): indexing information between mesh indexing declared in parallel and single core
    
    outputs:
        dfx_func_from_global(np.ndarray): array rearranged by single core indexing based on values calculated by parallel
        global_array(np.ndarray): gathered indexing array based on parallel
'''
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
'''
    inputs:
        dfx_func_from_global(np.ndarray): array rearranged by single core indexing based on values calculated by parallel
        global_array(np.ndarray): gathered indexing array based on parallel
        local_dof_size(numpy → list): gather local data
        serial_to_global(list): indexing information between mesh indexing declared in parallel and single core
    
    output:
        scatter_global_array(list → numpy): scattered rearranged by parallel indexing array from single core
'''
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