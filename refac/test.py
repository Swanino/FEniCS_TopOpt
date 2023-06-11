# PYTHON LIBRARIES ---------------------------------
from mpi4py import MPI

# PACKAGE LIBRARIES ---------------------------------
from dfx_topopt.fem.params import DimParam, MeshParams, ElasticParams, OptimParams
# from dfx_topopt.fem.elasticity import Elasticity
# from dfx_topopt.opts.topopt import TopOpt

'''
    Until Now:
        1. Subdividing which has been working on one file so far through parallel refactoring
        2. Inserting comment comments in each module file
        3. Confirm that there is no error up to the setting before calling the fem package
    ToDo:
        1. To check whether each package module runs without errors when called in the 'test.py'.
'''
def main(dim, nelx, nely, nelz, volfrac, penal, rmin, max_iter, ft, opt_solv):
    # WELCOME TO TOPOPT PROGRAM ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        print("\nMinimize compliance optimisation algorithm using HF\n")
    # RECALL PARAMETERS ---------------------------------
    dim_param = DimParam(dim)
    mesh_params = MeshParams(dim_param, nelx, nely, nelz, volfrac)
    elastic_params = ElasticParams(dim_param, penal)
    optim_params = OptimParams(rmin, max_iter, ft, opt_solv)

    if MPI.COMM_WORLD.rank == 0:
        print("ini. volume fraction: {0}, penalization power: {1:.3f}, min. filter radius: {2:.3f}".format(mesh_params.vf, elastic_params.penal, optim_params.rmin))
        print(["Sensitivity based", "Density based"][optim_params.ft] + " filtering")
        print(["OC method", "MMA"][optim_params.opt_solv] + " optimizer solver\n")

    # RECALL TOPOPT & ELASTICITY CLASSES ---------------------------------
    # optim = TopOpt(elastic_pars, optim_pars)
    # elas = Elasticity(elastic_pars)

    # DEFINE SUPPORT & LOAD ---------------------------------
    # elas.set_boundary_condition(elastic_pars)

    # SET UP THE VARIATIONAL PROBLEM AND SOLVER ---------------------------------
    # elas.setup_problem(optim_pars, optim.density)

    # FE-ANALYSIS ---------------------------------
    # elas.solve_problem()

# CALL MAIN FUNCTION ---------------------------------
if __name__ == "__main__":
    # DEFAULT INPUT PARAMETERS
    dim = 3
    nelx = 10
    nely = 10
    nelz = 10
    volfrac = 0.5
    penal = 3.0
    rmin = 3.0
    max_iter = 2000
    # ft == 0 -> sensitivity, ft == 1 -> density
    ft = 0
    # opt_solv == 0 -> OC, opt_solv == 1 -> MMA
    opt_solv = 0
    import sys
    if len(sys.argv)>1: dim      = int(sys.argv[1])
    if len(sys.argv)>2: nelx     = int(sys.argv[2])
    if len(sys.argv)>3: nely     = int(sys.argv[3])
    if len(sys.argv)>4: nelz     = int(sys.argv[4])
    if len(sys.argv)>5: volfrac  = float(sys.argv[5])
    if len(sys.argv)>6: penal    = float(sys.argv[6])
    if len(sys.argv)>7: rmin     = float(sys.argv[7])
    if len(sys.argv)>8: max_iter = int(sys.argv[8])
    if len(sys.argv)>9: ft       = int(sys.argv[9])
    if len(sys.argv)>10: opt_solv = int(sys.argv[10])
    main(dim, nelx, nely, nelz, volfrac, penal, rmin, max_iter, ft, opt_solv)