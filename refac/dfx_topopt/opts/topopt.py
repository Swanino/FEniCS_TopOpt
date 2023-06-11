# PYTHON LIBRARIES ---------------------------------
import numpy as np
from dolfinx import fem
from mpi4py import MPI

# OPTS LIBRARIES ---------------------------------
from ..fem.params import MeshParams, ElasticParams, OptimParams
from ..fem.elasticity import Elasticity
from ..fem.mesh import MeshGen
from ..par_src.par_utilities import mpi_serial_dof, mpi_gather, mpi_gathered_scatter
from .utilities import project_func
from .filters import helm_filter
from .solvers import oc_method

# TOPOLOGY OPTIMIZATION ---------------------------------
class TopOpt:
    '''
        TopOpt: class for topology optimisation
        __init__() prepares topology optimisation
        inputs:
            msh_params: MeshParams
            opt_params: OptimParams
            msh_info: MeshGen
    '''
    def __init__(self, msh_params: MeshParams, opt_params: OptimParams, msh_info: MeshGen) -> None:
        # ADJUST FILTER FADIUS FOR HELMHOLTZ FILTERING
        self.rmin_hf = np.divide(np.divide(opt_params.rmin, 2), np.sqrt(3))

        # CONDUCTOR MESH DOMAIN FUNCTION
        if MPI.COMM_WORLD.size > 1:
            if MPI.COMM_WORLD.rank == 1:
                self.density_con = fem.Function(msh_info.D0_con)
                self.nrs = self.density_con.vector.array.size
            else:
                self.density_con = None
                self.nrs = None

        # MESH DOMAIN FUNCTION
        # NODAL DESIGN VARIABLES
        self.den_node, self.sens_node, self.den_sens = fem.Function(msh_info.U1), fem.Function(msh_info.U1), fem.Function(msh_info.U1)
        # ELEMENT-WISE DESIGN VARIABLES
        self.density_old, self.density_new, self.density_opt = fem.Function(msh_info.D0), fem.Function(msh_info.D0), fem.Function(msh_info.D0)
        self.sensitivity, self.dv = fem.Function(msh_info.D0), fem.Function(msh_info.D0)
        # SET-UP INITIAL VALUES
        self.density_opt.vector.array = msh_params.vf
        self.dv.vector.array = 1.0

        # RELATION BETWEEN MESH AND CONDUCTOR MESH
        if MPI.COMM_WORLD.size > 1:
            self.size_global, self.gather_ranges, self.local_dof_size, self.serial_to_global = mpi_serial_dof(msh_info.msh, msh_info.msh_con_idx, self.density_opt, msh_info.D0)

    '''
        determined compliance (strain-energy)
        inputs:
            ela_params: ElasticParams
            ela_info: Elasticity
            msh_info: MeshGen
        output:
            objective(dolfinx.fem.Function): function space field
    '''
    def cal_comp(self, ela_params: ElasticParams, ela_info: Elasticity, msh_info: MeshGen):
        objective = project_func(self.density_opt**ela_params.penal * ela_info.psi(ela_info.u_sol), msh_info.D0)
        return objective

    '''
        projection process
        inputs:
            ela_params: ElasticParams
            ela_info: Elasticity
            msh_info: MeshGen
    '''
    def nodal_sens(self, ela_params: ElasticParams, ela_info: Elasticity, msh_info: MeshGen):
        # SENSITIVITY NODAL PROJECTION (DG0 TO CG1)
        self.den_node = project_func(self.density_opt, msh_info.C1)
        psi_node = project_func(ela_info.psi(ela_info.u_sol), msh_info.C1)
        self.sens_node.vector.array = -ela_params.penal * self.den_node.vector.array**(ela_params.penal-1) * psi_node.vector.array

    '''
        helmholtz filtering process
        inputs:
            opt_params: OptimParams
            msh_info: MeshGen
    '''
    def hf_filtering(self, opt_params: OptimParams, msh_info: MeshGen):
        # SENSITIVITY DISTANCE FILTERING
        if opt_params.ft == 0:
            # PREPARE DENSITY HELMHOLTZ FILTERING
            self.den_sens.vector.array = np.multiply(self.den_node.vector.array, self.sens_node.vector.array)
            # HELMHOLTZ FILTERING
            den_sens_til_node = helm_filter(self.den_sens, self.rmin_hf)
            # FILTERED HELMHOLTZ VARIABLE PROJECTION (CG1 TO DG0)
            density_til = project_func(den_sens_til_node, msh_info.D0)
            # SENSITIVITY ANALYSIS
            self.sensitivity.vector.array = np.divide(density_til.vector.array, np.maximum(1e-3, self.density_opt.vector.array))
        # DENSITY DISTANCE FILTERING
        elif opt_params.ft == 1:
            self.sensitivity = project_func(helm_filter(self.sens_node, self.rmin_hf), msh_info.D0)
            self.dv = project_func(helm_filter(project_func(self.dv, msh_info.C1), self.rmin_hf), msh_info.D0)

    '''
        Filtering on variables that have gone through the optimization algorithm
        inputs:
            opt_params: OptimParams
            msh_info: MeshGen
            den_new(np.ndarray): density variable after optimisation
    '''
    def var_filtering(self, opt_params: OptimParams, msh_info: MeshGen, den_new: np.ndarray):
        if opt_params.ft == 0:
            self.density_opt.vector.array = den_new
        elif opt_params.ft == 1:
            self.density_new.vector.array = den_new
            self.density_opt = project_func(helm_filter(project_func(self.density_new, msh_info.C1), self.rmin_hf), msh_info.D0)

    '''
    ---------- notice ----------
    The solvers written below are not self-parallelized functions,
    but are calculated by adjusting the degrees of freedom and indexing of variables before optimization calculation.
    '''
    '''
        optimality criteria optimisation method
        inputs:
            msh_params: MeshParams
            msh_info: MeshGen
        output:
            density_new(np.ndarray): variable after optimisation process
    '''
    def oc_optim(self, msh_params: MeshParams, msh_info: MeshGen) -> np.ndarray:
        if MPI.COMM_WORLD.size == 1:
            density_new = oc_method(0.0, 1e5, 0.2, 1e-3, msh_params.vf, msh_info.num_cells, self.density_opt.vector.array, self.sensitivity.vector.array, self.dv.vector.array)
        elif MPI.COMM_WORLD.size > 1:
            # GATHERING VARIABLES
            dens_gth, dens_glo = mpi_gather(self.density_opt, self.size_global, self.gather_ranges, self.serial_to_global)
            sens_gth, sens_glo = mpi_gather(self.sensitivity, self.size_global, self.gather_ranges, self.serial_to_global)
            dv_gth, dv_glo = mpi_gather(self.dv, self.size_global, self.gather_ranges, self.serial_to_global)
            if MPI.COMM_WORLD.rank == 1:
                density_new_con = oc_method(0.0, 1e5, 0.2, 1e-3, msh_params.vf, msh_info.num_cells, dens_gth, sens_gth, dv_gth)
            else:
                density_new_con = None
            density_new_con = MPI.COMM_WORLD.bcast(density_new_con, root=1)
            density_new = mpi_gathered_scatter(density_new_con, dens_glo, self.local_dof_size, self.serial_to_global)
        return density_new