import ufl
import numpy as np
import time
from dolfinx import fem, mesh, io
from mpi4py import MPI
from petsc4py import PETSc
from matplotlib import pyplot as plt
from util import *

# LINEAR ELASTICITY PROBLEM -------------------------------------------------``
class ElasticPars:
    def __init__(self, dim = 3, nel = (32, 32, 32), lmd=0.6, mu=0.4) -> None:
        self.dim = dim
        if dim == 2:    
            self.nelx, self.nely = nel
            self.nelz = 1
        elif dim == 3:
            self.nelx, self.nely, self.nelz = nel
        # material parameters 
        self.lmd = lmd # lambda 
        self.mu = mu # mu 
        self.lame_to_Ev()
    
    def lame_to_Ev(self):
        if self.dim == 2:
            self.E = self.mu * (3 * self.lmd + 2 * self.mu) / (self.lmd + self.mu)
            self.nu = self.lmd / (2 * (self.lmd + self.mu))
        elif self.dim == 3:
            self.E = 4.0*self.mu * (self.lmd + self.mu) / (self.lmd + 2.*self.mu)
            self.nu = self.lmd / (self.lmd + 2.*self.mu)

    def __str__(self) -> str:
        return f"ElasticPars(nelx={self.nelx}, nely={self.nely}, nelz={self.nelz}, lmd = {self.lmd}, mu = {self.mu}, E={self.E}, mu={self.mu})"
    
class Elasticity:
    '''
    Elasticity: class for finite element analysis
    __init__() prepares finite element analysis
    inputs
        elpars: ElasticPars
    '''
    def __init__(self, elpars: ElasticPars, 
                 petsc_options={"ksp_type": "preonly","pc_type": "lu","pc_factor_mat_solver_type": "mumps"})->None:
        # mesh generation --------------------------------
        self.elpars = elpars
        self.petsc_options = petsc_options
        self.dim = elpars.dim
        
        if self.dim == 2:
            self.msh = mesh.create_rectangle(MPI.COMM_WORLD, (np.zeros(2), [elpars.nelx, elpars.nely]), [elpars.nelx, elpars.nely], cell_type=mesh.CellType.quadrilateral, ghost_mode=mesh.GhostMode.shared_facet)
        elif self.dim == 3:
            self.msh = mesh.create_box(MPI.COMM_WORLD, [np.zeros(3), [elpars.nelx, elpars.nely, elpars.nelz]], [elpars.nelx, elpars.nely, elpars.nelz], cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)
        msh = self.msh 

        # function spaces ----------------------------
        self.U1 = fem.VectorFunctionSpace(msh, ("CG", 1)) # displacement basiss
        self.D0 = fem.FunctionSpace(msh, ("DG", 0)) # density
        self.u, self.v = ufl.TrialFunction(self.U1), ufl.TestFunction(self.U1) # note that this is ufl
        self.u_sol = fem.Function(self.U1) 

        print(f"{self.dim} dimensional problem, and mesh size of {elpars.nelx} x {elpars.nely} x {elpars.nelz} is created")
    
    def set_boundary_condition(self):
        # define support ---------------------------------
        print("currently supports left clamp problem only")
        def left_clamp(x): 
            return np.isclose(x[0], 0.0)
        f_dim = self.msh.topology.dim - 1 # facet dimension
        bc_facets = mesh.locate_entities_boundary(self.msh, f_dim, left_clamp) # boundary facets
        if self.dim == 2:
            u_zero = np.array([0., 0.], dtype=PETSc.ScalarType) # predefined displacement
        elif self.dim == 3:
            u_zero = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType) # predefined displacement
        print(fem.locate_dofs_topological(self.U1, f_dim, bc_facets))
        bc_l = fem.dirichletbc(u_zero, fem.locate_dofs_topological(self.U1, f_dim, bc_facets), self.U1)
        self.bcs = [bc_l]
        print(bc_l.dof_indices())


        # define load ---------------------------------
        print("currently supports point load only")
        if self.dim == 2:
            load_points = [(1, lambda x: x[0] == self.elpars.nelx)]
        elif self.dim == 3:
            load_points = [(1, lambda x: np.logical_and(x[0] == self.elpars.nelx, x[1] <= 2))]
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
        print(load_points)
        print(facet_indices)
        print(facet_markers)
        print(sorted_facets)

        ds = ufl.Measure("ds", domain=self.msh, subdomain_data=facet_tag) # measure for facet (surface)
        if self.dim == 2:
            self.f = ufl.dot(self.v, fem.Constant(self.msh, (0.0, -1.0))) * ds(1)
        elif self.dim == 3:
            self.f = ufl.dot(self.v, fem.Constant(self.msh, (0.0, -1.0, 0.0))) * ds(1)

    '''
        setup variational problem (linear elasticity)
    '''
    def setup_problem(self, density:fem.function.Function, penal:np.float64=3.0) -> None:
        sigma = lambda _u: 2.0 * self.elpars.mu * ufl.sym(ufl.grad(_u)) + self.elpars.lmd * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
        psi = lambda _u: self.elpars.lmd / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) + self.elpars.mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))

        k = ufl.inner(density**penal * sigma(self.u), ufl.grad(self.v)) * ufl.dx
        self.problem = fem.petsc.LinearProblem(k, self.f, bcs=self.bcs, petsc_options=self.petsc_options)
        # print(self.f)
        
        # viewer = PETSc.Viewer()
        # self.problem._A.assemble() # SHOULD NOT BE USED!!! 
        # self.problem._A.view(viewer.ASCII("./savem/testK0.vtu"))
        # self.problem.A.view(viewer.ASCII("./savem/testK.vtu"))
        # viewer.destroy()

    def solve_problem(self):
        # Should support any PETSC solver
        # However, not tested yet.
        print(self.problem._A.isAssembled())

        self.u_sol = self.problem.solve() 
        viewer = PETSc.Viewer()
        print(self.problem._A.isAssembled())
        self.problem._A.view(viewer.createASCII("./savem/testK.vtu"))
        # m = viewer.create(comm=MPI.COMM_WORLD)
        # m.setType(PETSc.Viewer.Type.EXODUSII)
        # m.setFileName("./savem/testKm.vtu")
        # self.problem._A.view(m)
        # self.problem._A.view(viewer.ASCII("./savem/testK.vtu"))
        # self.problem.A.view(viewer.createVTK("./savem/testK2.vtu", comm=MPI.COMM_WORLD))
        m  = viewer.createVTK("./savem/testK2.vtu", comm=MPI.COMM_WORLD)
        viewer.createVTK("./savem/testK2.vtu").view(self.problem.A)
        viewer.destroy()

    def export_to_vtk(self):
        # export to vtk
        vtkfile = fem.File("output/displacement.pvd")
        vtkfile << self.u_sol

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


if __name__ == "__main__":
    # test
    elpars = ElasticPars(dim=2, nel=(10,10))
    el = Elasticity(elpars)
    el.set_boundary_condition()
    density = fem.Function(el.D0) # if density \in C1, power is not defined
    density.x.array[:] = 1.0
    penal = 3.0
    el.setup_problem(density=density, penal=penal)
    el.solve_problem()
    viewer = PETSc.Viewer(comm=MPI.COMM_WORLD)
    # viewer(el.problem._A)
    # viewer(el.problem.b)



    msh = el.msh
    u_sol = el.u_sol

    # import mesh_pyvista
    # mesh_pyvista.plot_warped(msh, u_sol, warp_factor=2.0)

        # with io.VTXWriter(MPI.COMM_WORLD, "test.bp", u0) as f:
    #     f.write(0.0)
    # not working for some reason... ADIOS2 bug?


    with io.XDMFFile(MPI.COMM_WORLD, "./save/test.xdmf", "w") as f:
        f.write_mesh(msh)
        f.write_function(u_sol)

    viewer = PETSc.Viewer()
    u_sol.vector.view(viewer.ASCII("./savem/testq.vtu"))

    viewer.destroy()

    # import mesh_pyvista
    # mesh_pyvista.plot_warped(msh, u_sol, warp_factor=2.0)

    