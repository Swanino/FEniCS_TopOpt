from dolfin import *
from matplotlib import pyplot as plt
import numpy as np, sklearn.metrics.pairwise as sp
import os, psutil, time
# SET START TIME ---------------------------------
start = time.time()
# A 55 LINE TOPOLOGY OPTIMIZATION CODE ---------------------------------
def main(nelx, nely, nelz, volfrac, penal, rmin):
    sigma = lambda _u: 2.0 * mu * sym(grad(_u)) + lmbda * tr(sym(grad(_u))) * Identity(len(_u))
    psi = lambda _u: lmbda / 2 * (tr(sym(grad(_u))) ** 2) + mu * tr(sym(grad(_u)) * sym(grad(_u)))
    xdemf = XDMFFile("output/density.xdmf")
    mu, lmbda = Constant(0.4), Constant(0.6)
    # PREPARE FINITE ELEMENT ANALYSIS ---------------------------------
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(nelx, nely, nelz), nelx, nely, nelz)
    U = VectorFunctionSpace(mesh, "P", 1)
    D = FunctionSpace(mesh, "DG", 0)
    u, v = TrialFunction(U), TestFunction(U)
    u_sol, density_old, density = Function(U), Function(D), Function(D, name="density")
    density.vector()[:] = volfrac
    # DEFINE SUPPORT ---------------------------------
    support = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=1e-14)
    bcs = [DirichletBC(U,Constant((0.0, 0.0, 0.0)), support)]
    # DEFINE LOAD ---------------------------------
    load_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    CompiledSubDomain("x[0]==l && x[1]<=2", l=nelx).mark(load_marker, 1)
    ds = Measure("ds")(subdomain_data=load_marker)
    F = dot(v, Constant((0.0, -1.0, 0.0))) * ds(1)
    # SET UP THE VARIATIONAL PROBLEM AND SOLVER ---------------------------------
    K = inner(density**penal * sigma(u), grad(v)) * dx
    problem = LinearVariationalProblem(K, F, u_sol, bcs=bcs)
    solver = LinearVariationalSolver(problem)
    # PREPARE DISTANCE MATRICES FOR FILTER ---------------------------------
    midpoint = [cell.midpoint().array()[:] for cell in cells(mesh)]
    distance_mat = rmin - sp.euclidean_distances(midpoint, midpoint)
    distance_mat[distance_mat < 0] = 0
    distance_sum = distance_mat.sum(1)
    # START ITERATION ---------------------------------
    loop, change = 0, 1
    x_cord, y_cord = [], []
    while change > 0.01 and loop < 2000:
        loop = loop + 1
        density_old.assign(density)
        # FE-ANALYSIS ---------------------------------
        solver.solve()
        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS ---------------------------------
        objective = project(density**penal * psi(u_sol), D).vector()[:]
        sensitivity = -penal * (density.vector()[:])**(penal-1) * project(psi(u_sol), D).vector()[:]
        # FILTERING/MODIFICATION OF SENSITIVITIES ---------------------------------
        sensitivity = np.divide(distance_mat @ np.multiply(density.vector()[:], sensitivity), np.multiply(density.vector()[:], distance_sum))
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD ---------------------------------
        l1, l2, move = 0, 100000, 0.2
        while l2 - l1 > 1e-4:
            l_mid = 0.5 * (l2 + l1)
            density_new = np.maximum(0.001, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-sensitivity/l_mid)))))
            l1, l2 = (l_mid, l2) if sum(density_new) - volfrac*mesh.num_cells() > 0 else (l1, l_mid)
        density.vector()[:] = density_new
        # CHECK CPU/RAM MEMORY ---------------------------------
        pid = os.getpid()
        py = psutil.Process(pid)
        cpu_usage = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
        cpu_usage = cpu_usage.replace("\n", "")
        ram_memory_usage = round(py.memory_info()[0] / 2. ** 30, 2)
        # PRINT RESULTS ON THE COMMAND SCREEN ---------------------------------
        change = norm(density.vector() - density_old.vector(), norm_type="linf", mesh=mesh)
        print("it.: {0} , obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, sum(objective), sum(density.vector()[:]) / mesh.num_cells(), change))
        print("CPU usage:", cpu_usage, "%")
        print("RAM usage:", ram_memory_usage, "%\n")
        xdemf.write(density, loop)
        # PLOT OBJECTIVE FUNCTION ---------------------------------
        x_cord.append(loop)
        y_cord.append(sum(objective))
    plt.plot(x_cord, y_cord, color='b')
    plt.scatter(x_cord, y_cord, marker='.', color='b')
    # FIND MAXIMUM VALUE IN CHART ---------------------------------
    max_id1 = np.argmax(y_cord)
    plt.plot(x_cord[max_id1], y_cord[max_id1], marker='s', color='r', label='max value: {0}'.format(round(y_cord[max_id1], 2)), ms=10)
    # SHOW X/Y COORDINATES & SUB-LINE ---------------------------------
    plt.xlabel("iteration")
    plt.ylabel("compliance")
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)
    # SAVE CHART ---------------------------------
    plt.savefig('output/objective_function.jpg')
    # CHECK END TIME & SHOW RUNNING TIME ---------------------------------
    end = time.time()
    print("RUNNING Time:", end - start, "sec")
    plt.show()