from dolfinx import mesh
from mpi4py import MPI
from ProblemDef import TopOptProblem
import numpy as np
from dolfinx import fem, mesh, io
from petsc4py import PETSc
import ufl

# mesh generation --------------------------------
class Cantilever2D(TopOptProblem):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 2
        self.lx, self.ly = 160,40
        self.nelx, self.nely = 160,40

        self.define_mesh()
        self.BC_facets, self.u_hat = self.define_BC()
        self.LC_facet_tag, self.mag_force = self.define_LC()

    def define_mesh(self):
        self.msh = mesh.create_box(MPI.COMM_WORLD, (np.zeros(2), [self.lx, self.ly]), [self.nelx, self.nely], cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)

    def define_BC(self):
        # define support ---------------------------------
        print("currently supports left clamp problem only")
        def left_clamp(x): 
            return np.isclose(x[0], 0.0)
        
        f_dim = self.msh.topology.dim - 1 # facet dimension
        bc_facets = mesh.locate_entities_boundary(self.msh, f_dim, left_clamp) # boundary facets
        u_zero = np.array([0., 0.], dtype=PETSc.ScalarType) # predefined displacement
        return bc_facets, u_zero

    def define_LC(self):
        print("currently supports point load only")
        load_points = [(1, lambda x: x[0] == self.elpars.nelx)]
        facet_indices, facet_markers = [], []
        f_dim = self.msh.topology.dim - 1
        for (marker, locator) in load_points:
            facets = mesh.locate_entities(self.msh, f_dim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full(len(facets), marker))
        facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
        facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = mesh.meshtags(self.msh, f_dim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        mag_force = -1.0 
        return facet_tag, mag_force 
    
    def __str__(self) -> str:
        return f"Cantilever2D: lx={self.lx}, ly={self.ly}, nelx={self.nelx}, nely={self.nely}"
    
    
