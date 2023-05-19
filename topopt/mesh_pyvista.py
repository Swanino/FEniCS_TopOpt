from dolfinx import *
from dolfinx.fem import *
from dolfinx.mesh import *
import pyvista
from dolfinx import *

import elasticity as elas
import numpy as np

# elpars = elas.ElasticPars(dim=3, nel=(32, 32, 32), lmd=0.6, mu=0.4)

# fea = elas.Elasticity(elpars)
# fea.set_boundary_condition()
# density = fem.Function(fea.D0) # if density \in C1, power is not defined
# density.x.array[:] = 1.0
# penal = 3.0
# fea.setup_problem(density=density, penal=3.0)


# fea.solve_problem()

# Plotting

def plot_warped(msh, u, warp_factor=2.0):
    cells, types, x = plot.create_vtk_mesh(msh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    print(u._V.dofmap.index_map_bs)
    grid.point_data["u"] = u.x.array.reshape(x.shape[0], u._V.dofmap.index_map_bs)

    grid.set_active_scalars("u")
    warped = grid.warp_by_vector("u", warp_factor=2)

    subplotter = pyvista.Plotter(shape=(1, 1))
    subplotter.subplot(0, 0)
    subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(warped, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()
    subplotter.show()