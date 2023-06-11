# PYTHON LIBRARIES ---------------------------------
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

# PACKAGE LIBRARIES ---------------------------------
from .params import DimParam, MeshParams

# LINEAR ELASTICITY PROBLEM ---------------------------------
class MeshGen:
    '''
        MeshGen: class for dolfinx mesh generation
        __init__() prepares mesh generation
        inputs:
            dim_param: DimParam
            msh_params: MeshParams
    '''
    def __init__(self, dim_param: DimParam, msh_params: MeshParams) -> None:
        # MESH ON CONDUCTOR CORE
        if MPI.COMM_WORLD.size > 1:
            if MPI.COMM_WORLD.rank == 1:
                if dim_param.dime == 2:
                    self.msh_con = mesh.create_rectangle(MPI.COMM_SELF, [np.zeros(2), [msh_params.nlx, msh_params.nly]], [msh_params.nlx, msh_params.nly], cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)
                if dim_param.dime == 3:
                    self.msh_con = mesh.create_box(MPI.COMM_SELF, [np.zeros(3), [msh_params.nlx, msh_params.nly, msh_params.nlz]], [msh_params.nlx, msh_params.nly, msh_params.nlz], cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)
                msh_con_idx = self.msh_con.topology.original_cell_index
                self.num_cells_con = self.msh_con.topology.index_map(self.msh_con.topology.dim).size_local  # number of conductor mesh cells

                # CONDUCTOR MESH DOMAIN FUNCTION SPACES
                self.D0_con = fem.FunctionSpace(self.msh_con, ("DG", 0))
            else:
                msh_con_idx = None
                self.num_cells_con = None
            self.msh_con_idx = MPI.COMM_WORLD.bcast(msh_con_idx, root=1)

        # MESH GENERATION
        if dim_param.dime == 2:
            self.msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.zeros(2), [msh_params.nlx, msh_params.nly]], [msh_params.nlx, msh_params.nly], cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)
        elif dim_param.dime == 3:
            self.msh = mesh.create_box(MPI.COMM_WORLD, [np.zeros(3), [msh_params.nlx, msh_params.nly, msh_params.nlz]], [msh_params.nlx, msh_params.nly, msh_params.nlz], cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)
        self.t_dim = self.msh.topology.dim                                      # cell dimension
        self.f_dim = self.msh.topology.dim - 1                                  # facet dimension
        self.num_cells = self.msh.topology.index_map(self.t_dim).size_local     # number of mesh cells

        # MESH DOMAIN FUNCTION SPACES
        self.U1 = fem.VectorFunctionSpace(self.msh, ("CG", 1))                  # displacement basis
        self.D0 = fem.FunctionSpace(self.msh, ("DG", 0))                        # element-wise basis
        self.C1 = fem.FunctionSpace(self.msh, ("CG", 1))                        # nodal basis