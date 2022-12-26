from dolfin import *
from mpi4py import MPI
from matplotlib import pyplot as plt
import numpy as np
import os, psutil, time
# HELMHOLTZ WEAK FORM PDE FILTER FUNCTION ---------------------------------
def helm_filter(rho_n, r_min, annotate=False, name="Filtered"):
    V = rho_n.function_space()
    rho, w = TrialFunction(V), TestFunction(V)
    a = (r_min**2) * inner(grad(rho), grad(w)) * dx + rho * w * dx
    L = rho_n * w * dx
    A, b = assemble_system(a, L)
    rho = Function(V, name=name)
    solve(A, rho.vector(), b)
    return rho
# SET START TIME ---------------------------------
start = time.time()
# A 55 LINE TOPOLOGY OPTIMIZATION CODE (IN 3D) ---------------------------------
def main(nelx, nely, nelz, volfrac, penal, rmin):
    # CHECK PATH ---------------------------------
    path = './output'
    if not os.path.isdir(path):
        os.makedirs(path)
        print("The output folder did not exist and was created.")
    rmin = np.divide(np.divide(rmin, 2), np.sqrt(3))
    sigma = lambda _u: 2.0 * mu * sym(grad(_u)) + lmbda * tr(sym(grad(_u))) * Identity(len(_u))
    psi = lambda _u: lmbda / 2 * (tr(sym(grad(_u))) ** 2) + mu * tr(sym(grad(_u)) * sym(grad(_u)))
    xdemf = XDMFFile("output/density.xdmf")
    xdemf_sens = XDMFFile("output/sens.xdmf")
    mu, lmbda = Constant(0.4), Constant(0.6)
    # PREPARE FINITE ELEMENT ANALYSIS ---------------------------------
    mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(nelx, nely, nelz), nelx, nely, nelz)
    U1 = VectorFunctionSpace(mesh, "CG", 1)
    C1 = FunctionSpace(mesh, "CG", 1)
    D0 = FunctionSpace(mesh, "DG", 0)
    u, v = TrialFunction(U1), TestFunction(U1)
    u_sol, density_old, density = Function(U1), Function(D0), Function(D0)
    den_node, den_sens = Function(C1), Function(C1)
    sens = Function(D0)
    density.vector()[:] = volfrac
    # DEFINE SUPPORT ---------------------------------
    support = CompiledSubDomain("near(x[0], 0.0, tol) && on_boundary", tol=1e-14)
    bcs = [DirichletBC(U1, Constant((0.0, 0.0, 0.0)), support)]
    # DEFINE LOAD ---------------------------------
    load_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    CompiledSubDomain("x[0]==l && x[1]<=2", l=nelx).mark(load_marker, 1)
    ds = Measure("ds")(subdomain_data=load_marker)
    F = dot(v, Constant((0.0, -1.0, 0.0))) * ds(1)
    # SET UP THE VARIATIONAL PROBLEM AND SOLVER ---------------------------------
    K = inner(density**penal * sigma(u), grad(v)) * dx
    problem = LinearVariationalProblem(K, F, u_sol, bcs=bcs)
    solver = LinearVariationalSolver(problem)
    # OPEN RESULTS TEXT FILE ---------------------------------
    # results_txt = open('output/results.txt', 'w')
    # START ITERATION ---------------------------------
    loop, change = 0, 1
    x_cord, y_cord, y_cord_net, sensi = [], [], [], []
    while change > 0.01 and loop < 2000:
        loop = loop + 1
        density_old.assign(density)
        # FE-ANALYSIS ---------------------------------
        solver.solve()
        print(u_sol.vector()[:])
        # OBJECTIVE FUNCTION ---------------------------------
        objective = project(density**penal * psi(u_sol), D0).vector()[:]
        # SENSITIVITY NODAL PROJECTION (DG0 TO CG1) ---------------------------------
        den_node.interpolate(project(density, C1))
        sens_node = -penal * (den_node.vector()[:])**(penal-1) * project(psi(u_sol), C1).vector()[:]
        # PREPARE DENSITY HELMHOLTZ FILTERING ---------------------------------
        den_sens.vector()[:] = np.multiply(den_node.vector()[:], sens_node)
        # HELMHOLTZ FILTERING ---------------------------------
        helm_til_node = helm_filter(den_sens, rmin, annotate=False)
        # FILTERED HELMHOLTZ VARIABLE PROJECTION (CG1 TO DG0) ---------------------------------
        density_til = Function(D0)
        density_til.interpolate(project(helm_til_node, D0))
        # SENSITIVITY ANALYSIS ---------------------------------
        sens_np = np.divide(density_til.vector()[:], np.maximum(1e-3, density.vector()[:]))
        sens.vector()[:] = np.divide(density_til.vector()[:], np.maximum(1e-3, density.vector()[:]))
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD ---------------------------------
        l1, l2, move = 0, 1e5, 0.2
        while l2 - l1 > 1e-4:
            l_mid = 0.5 * (l2 + l1)
            if (-sens_np/l_mid <= 0).any(): break
            density_new = np.maximum(0.001, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-sens_np/l_mid)))))
            l1, l2 = (l_mid, l2) if sum(density_new) - volfrac*mesh.num_cells() > 0 else (l1, l_mid)
        density.vector()[:] = density_new
        # CHECK CPU/RAM MEMORY ---------------------------------
        pid = os.getpid()
        py  = psutil.Process(pid)
        cpu_usage = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
        cpu_usage = cpu_usage.replace("\n", "")
        ram_memory_usage = round(py.memory_info()[0] / 2. ** 30, 2)
        # PRINT RESULTS ON THE COMMAND SCREEN ---------------------------------
        change = norm(density.vector() - density_old.vector(), norm_type="linf", mesh=mesh)
        print("it.: {0} , obj.: {1:.3f}, Vol.: {2:.3f}, ch.: {3:.3f}, Pros.: {4} on {5}".format(loop, sum(objective), sum(density.vector()[:]) / mesh.num_cells(), change, MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size))
        print("CPU usage:", cpu_usage, "%", "in {0} on {1}".format(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size))
        print("RAM usage:", ram_memory_usage, "%", "in {0} on {1}\n".format(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size))
        xdemf.write(density, loop)
        xdemf_sens.write(sens, loop)
        # PLOT OBJECTIVE FUNCTION ---------------------------------
        x_cord.append(loop)
        y_cord.append(sum(objective))
        y_cord_net.append(mesh.mpi_comm().allreduce(sum(objective), op=MPI.SUM))
        sensi.append(sum(sens_np))
    # PLOT EACH MPI PROCESSOR COMPLIANCE VALUE ---------------------------------
    plt.figure(1)
    plt.plot(x_cord, y_cord, color='b')
    plt.scatter(x_cord, y_cord, marker='.', color='b')
    # FIND MAX/MINIMUM VALUE IN CHART ---------------------------------
    max_id1 = np.argmax(y_cord)
    min_id1 = np.argmin(y_cord)
    plt.plot(x_cord[max_id1], y_cord[max_id1], marker='s', color='r', label='max value: {0}'.format(round(y_cord[max_id1], 2)), ms=5)
    plt.plot(x_cord[min_id1], y_cord[min_id1], marker='s', color='r', label='min value: {0}'.format(round(y_cord[min_id1], 2)), ms=5)
    # SHOW X/Y COORDINATES & SUB-LINE ---------------------------------
    plt.xlabel("iteration")
    plt.ylabel("compliance")
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)
    # SAVE CHART ---------------------------------
    plt.savefig('output/objective_function_processor.jpg')
    # PLOT NET MPI COMPLIANCE VALUE USING MPI GATHER ---------------------------------
    if MPI.COMM_WORLD.rank == 0:
        plt.figure(2)
        plt.plot(x_cord, y_cord_net, color='r')
        plt.scatter(x_cord, y_cord_net, marker='.', color='r')
        # FIND MAXIMUM VALUE IN CHART ---------------------------------
        max_id2 = np.argmax(y_cord_net)
        min_id2 = np.argmin(y_cord_net)
        plt.plot(x_cord[max_id2], y_cord_net[max_id2], marker='s', color='c', label='max value: {0}'.format(round(y_cord_net[max_id2], 2)), ms=5)
        plt.plot(x_cord[min_id2], y_cord_net[min_id2], marker='s', color='c', label='min value: {0}'.format(round(y_cord_net[min_id2], 2)), ms=5)
        # SHOW X/Y COORDINATES & SUB-LINE ---------------------------------
        plt.xlabel("iteration")
        plt.ylabel("net compliance")
        plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
        plt.legend(fontsize=10)
        # SAVE CHART ---------------------------------
        plt.savefig('output/objective_function_mpi_net.jpg')
    count = 1
    if os.path.isdir('./output/objective_function_mpi_net.jpg'):
        plt.savefig('output/objective_function_mpi_net_({0})'.format(count))
        count+=1
    # PLOT EACH MPI PROCESSOR COMPLIANCE VALUE ---------------------------------
    plt.figure(3)
    plt.plot(x_cord, sensi, color='c')
    plt.scatter(x_cord, sensi, marker='.', color='c')
    # FIND MAXIMUM VALUE IN CHART ---------------------------------
    max_id3 = np.argmax(sensi)
    min_id3 = np.argmin(sensi)
    plt.plot(x_cord[max_id3], sensi[max_id3], marker='s', color='r', label='max value: {0}'.format(round(sensi[max_id3], 2)), ms=5)
    plt.plot(x_cord[min_id3], sensi[min_id3], marker='s', color='r', label='min value: {0}'.format(round(sensi[min_id3], 2)), ms=5)
    # SHOW X/Y COORDINATES & SUB-LINE ---------------------------------
    plt.xlabel("iteration")
    plt.ylabel("sensitivity")
    plt.grid(True, axis='y', color='red', alpha=0.5, linestyle='--')
    plt.legend(fontsize=10)
    # CHECK END TIME & SHOW RUNNING TIME ---------------------------------
    end = time.time()
    print("RUNNING Time:", end-start, "sec")
    plt.show()