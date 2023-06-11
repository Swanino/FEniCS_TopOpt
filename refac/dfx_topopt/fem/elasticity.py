# PYTHON LIBRARIES ---------------------------------
import numpy as np
import ufl
from dolfinx import fem, mesh
from petsc4py import PETSc

# PACKAGE LIBRARIES ---------------------------------
from .params import DimParam, MeshParams, ElasticParams
from .mesh import MeshGen

# LINEAR ELASTICITY PROBLEM ---------------------------------
class Elasticity:
    '''
        Elasticity: class for finite element analysis
        __init__() prepares finite element analysis
        inputs:
            ela_params: ElasticParams
    '''
    def __init__(self, msh_params: MeshParams, ela_params: ElasticParams, msh_info: MeshGen) -> None:
        # DECLARE HYPER ELASTICITY FUNCTIONS
        self.sigma = lambda _u: 2.0 * ela_params.meu * ufl.sym(ufl.grad(_u)) + ela_params.lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
        self.psi = lambda _u: ela_params.lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + ela_params.meu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))

        # INSTANCE CHARACTERS
        self.bcs = None
        self.f = None

        # MESH DOMAIN FUNCTION
        self.u, self.v = ufl.TrialFunction(msh_info.U1), ufl.TestFunction(msh_info.U1)      # note that these are ufl
        self.u_sol = fem.Function(msh_info.U1)
        self.density_ini = fem.Function(msh_info.D0)
        self.density_ini.vector.array = msh_params.vf

    '''
        set boundary conditions and load (points)
        inputs:
            dim_param: DimParam
            msh_param: MeshParams
            msh_info: MeshGen
    '''
    def set_boundary_condition(self, dim_param: DimParam, msh_params: MeshParams, msh_info: MeshGen):
        # DEFINE SUPPORT
        print("currently supports left clamp problem only")
        def left_clamp(x):
            return np.isclose(x[0], 0.0)

        bc_facets = mesh.locate_entities_boundary(msh_info.msh, msh_info.f_dim, left_clamp)  # boundary facets
        if dim_param.dime == 2:
            u_zero = np.array([0.0, 0.0], dtype=PETSc.ScalarType)                            # predefined displacement
        elif dim_param.dime == 3:
            u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)                       # predefined displacement
        bc_l = fem.dirichletbc(u_zero, fem.locate_dofs_topological(msh_info.U1, msh_info.f_dim, bc_facets), msh_info.U1)
        self.bcs = [bc_l]

        # DEFINE LOAD
        print("currently supports point load only")
        load_points = [(1, lambda x: np.logical_and(x[0] == msh_params.nlx, x[1] <= 2))]
        facet_indices, facet_markers = [], []
        for (marker, locator) in load_points:
            facets = mesh.locate_entities(msh_info.msh, msh_info.f_dim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(msh_info.msh, msh_info.f_dim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        ds = ufl.Measure("ds", domain=msh_info.msh, subdomain_data=facet_tag)  # measure for facet (surface)
        if dim_param.dime == 2:
            self.f = ufl.dot(self.v, fem.Constant(msh_info.msh, (0.0, -1.0))) * ds(1)
        elif dim_param.dime == 3:
            self.f = ufl.dot(self.v, fem.Constant(msh_info.msh, (0.0, -1.0, 0.0))) * ds(1)

    '''
        set-up variational problem (linear elasticity)
        inputs:
            ela_params: ElasticParams
            petsc_option
    '''
    def setup_problem(self, ela_params: ElasticParams, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}):
        k = ufl.inner(self.density_ini**ela_params.penal * self.sigma(self.u), ufl.grad(self.v)) * ufl.dx
        self.problem = fem.petsc.LinearProblem(k, self.f, bcs=self.bcs, petsc_options=petsc_options)

    '''
        solve the variational problem (fea)
    '''
    def solve_problem(self):
        # SHOULD SUPPORT ANY PETSC SOLVER
        # HOWEVER, NOT TESTED YET.
        self.u_sol = self.problem.solve()