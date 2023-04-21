import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from util import *


# LINEAR ELASTICITY PROBLEM -------------------------------------------------
class ElasticPars:
    def __init__(self, nelx=32, nely=32, nelz=32, lmd=0.6, mu=0.4) -> None:
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.lmd = lmd
        self.mu = mu

    def __str__(self) -> str:
        return f"ElasticPars(nelx={self.nelx}, nely={self.nely}, nelz={self.nelz}, E={self.E}, mu={self.mu})"
    
class Elasticity:
    '''
    Elasticity: class for finite element analysis
    __init__() prepares finite element analysis
    inputs
        elpars: ElasticPars
    '''
    def __init__(self, elpars: ElasticPars)->None:
        # mesh generation --------------------------------
        self.msh = mesh.create_box(MPI.COOM_WORLD, [np.zeros(3), [elpars.nelx, elpars.nely, elpars.nelz]], [elpars.nelx, elpars.nely, elpars.nelz], cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)
        msh = self.msh # alias

        # function spaces ----------------------------
        V = fem.FunctionSpace(mesh, ("CG", 1)) # isoparametric basis function
        u = fem.Function(V)                    # displacement
        v = fem.TestFunction(V)                # test function

        U1 = fem.VectorFunctionSpace(msh, ("CG", 1)) # displacement basiss
        C1 = fem.FunctionSpace(msh, ("CG", 1)) # isoparametric basis function
        D0 = fem.FunctionSpace(msh, ("DG", 0)) # element-wise constant function space
        u, v = ufl.TrialFunction(U1), ufl.TestFunction(U1)
        u_sol, density_old, density = fem.Function(U1), fem.Function(D0), fem.Function(D0)
        den_node, den_sens = fem.Function(C1), fem.Function(C1)


        density.x.array[:] = volfrac



        # define support ---------------------------------
        print("currently supports left clamp problem only")
        def left_clamp(x): 
            return np.isclose(x[0], 0.0)
        f_dim = msh.topology.dim - 1 # facet dimension
        bc_facets = mesh.locate_entities_boundary(msh, f_dim, left_clamp) # boundary facets
        u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType) # predefined displacement
        bc_l = fem.dirichletbc(u_zero, fem.locate_dofs_topological(U1, f_dim, bc_facets), U1)
        bcs = [bc_l]
        # define load ---------------------------------
        load_points = [(1, lambda x: np.logical_and(x[0] == nelx, x[1] <= 2))]
        facet_indices, facet_markers = [], []
        f_dim = msh.topology.dim - 1
        for (marker, locator) in load_points:
            facets = mesh.locate_entities(msh, f_dim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(msh, f_dim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)
        f = ufl.dot(v, fem.Constant(msh, (0.0, -1.0, 0.0))) * ds(1)

        return mesh, bc

    '''
        forward_analysis: forward finite element analysis
        inputs
            elpars: ElasticPars
            isprint: bool, whether to print mesh to xdmf file
            foldername: str, foldername to print mesh to
    '''
    def forward_analysis(self, elpars: ElasticPars, isprint: bool = True, foldername: str = "output"):
        # cast to scalar type
        mu = PETSc.ScalarType(elpars.mu)
        lmd  = PETSc.ScalarType(elpars.lmd)
        
        if isprint:
            from pathlib import Path
            Path(foldername).mkdir(parents=True, exist_ok=True)
            fname = print_mesh(mesh)
            print(f"mesh printed to {fname}")


        sigma = lambda _u: 2.0 * mu * ufl.sym(ufl.grad(_u)) + lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
        psi = lambda _u: lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))

