# PYTHON LIBRARIES ---------------------------------
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

# DIMENSION PARAMETER ---------------------------------
class DimParam:
    '''
        DimParam: class which receives dimension parameter
        __init__() receives mesh dimension parameter
        inputs:
            dime(int): number of dimension who chooses '2' or '3'
    '''
    def __init__(self, dime: int) -> None:
        if MPI.COMM_WORLD.rank == 0:
            print("The user-selected mesh dimension is '{0}'".format(dime))
        else:
            dime = None
        MPI.COMM_WORLD.barrier()
        self.dime = MPI.COMM_WORLD.bcast(dime, root=0)

    def __str__(self) -> str:
        return f"DimParam(dim = {self.dime})"

# MESH PARAMETERS ---------------------------------
class MeshParams:
    '''
        MeshParams: class which variables needed to create dolfinx mesh
        __init__() receives mesh parameter
        inputs:
            dim_param: DimParam
            nelx(int): number of element along to x dir.
            nely(int): number of element along to y dir.
            nelz(int): number of element along to z dir.
            volfrac(np.float): volume fraction of mesh domain
    '''
    def __init__(self, dim_param: DimParam, nelx: int, nely: int, nelz: int, volfrac: np.float) -> None:
        # IN 2D CASE
        if dim_param.dime == 2:
            self.nlx = nelx
            self.nly = nely
            self.nlz = None
            if MPI.COMM_WORLD.rank == 0:
                print("sizes: {0} X {1}, parallel core number is {2}".format(self.nlx, self.nly, MPI.COMM_WORLD.size))
        # IN 3D CASE
        elif dim_param.dime == 3:
            self.nlx = nelx
            self.nly = nely
            self.nlz = nelz
            if MPI.COMM_WORLD.rank == 0:
                print("Mesh sizes: {0} X {1} X {2}, parallel core number is {3}".format(self.nlx, self.nly, self.nlz, MPI.COMM_WORLD.size))
        # VOLUME FRACTION PARAMETER
        self.vf = volfrac

    def __str__(self) -> str:
        return f"MeshParams(nlx = {self.nlx}, nly = {self.nly}, nlz = {self.nlz}, vf = {self.vf})"

# LINEAR ELASTICITY PARAMETERS ---------------------------------
class ElasticParams:
    '''
        ElasticParams: class which variables needed to determine finite element elasticity problem
        __init__() receives parameter associated with elasticity
        inputs:
            dim_param: DimParam
            penal(np.float): SIMP method penalty factor
    '''
    def __init__(self, dim_param: DimParam, penal) -> None:
        self.penal = penal
        # MATERIAL PARAMETERS
        self.lmd = PETSc.ScalarType(0.6)    # lambda: 0.6
        self.meu = PETSc.ScalarType(0.4)    # mu: 0.4

        # YOUNG'S MOD AND POISSON'S RATIO (LAMÉ'S PARAMETERS)
        if dim_param.dime == 2:
            self.E = self.meu * (3 * self.lmd + 2 * self.meu) / (self.lmd + self.meu)
            self.nu = self.lmd / (2 * (self.lmd + self.meu))
        elif dim_param.dime == 3:
            self.E = 4.0 * self.meu * (self.lmd + self.meu) / (self.lmd + 2 * self.meu)
            self.nu = self.lmd / (self.lmd + 2 * self.meu)

    def __str__(self) -> str:
        return f"ElasticParams(penal = {self.penal}, lmd = {self.lmd}, _mu = {self.meu}, E = {self.E}, nu = {self.nu})"

# OPTIMIZATION PARAMETERS ---------------------------------
class OptimParams:
    '''
        OptimParams: class which variables related to optimising process
        __init__() receives parameter
        inputs:
            rmin(np.float): minimum radius of filter
            max_iter(int): maximum number of optimisation iteration
            ft(int): flag of filtering method (density filtering or sensitivity filtering)
            opt_solv(int): flag of optimising method
    '''
    def __init__(self, rmin, max_iter, ft, opt_solv) -> None:
        self.rmin = rmin
        self.max_iter = max_iter
        self.ft = ft
        self.opt_solv = opt_solv

        # WARNING MESSAGE RELATED TO HELMHOLTZ FILTER RADIUS
        if rmin < 3.0:
            print(f"In Helmholtz filtering, 'rmin' should be greater than 3.0")
            self.rmin = 3.0

        # WARNING MESSAGE RELATED TO MAXIMUM ITERATION OF TOPOLOGY OPTIMISATION
        if max_iter > 2000:
            print(f"Usually, 'max_iter' does not exceed 2,000")
            self.max_iter = 2000

    def __str__(self) -> str:
        return f"OptimParams(rmin = {self.rmin}, max_iter = {self.max_iter}, ft = {self.ft}, opt_solv = {self.opt_solv})"