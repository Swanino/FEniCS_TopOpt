from dolfinx import *
from dolfinx.fem import * #Function, FunctionSpace, create_unit_square
from dolfinx.mesh import *
import pyvista
import numpy as np
from mpi4py import MPI

def plot_scalar():

    # We start by creating a unit square mesh and interpolating a function
    # into a first order Lagrange space
    msh = create_unit_square(MPI.COMM_WORLD, 12, 12, cell_type=CellType.quadrilateral)
    V = FunctionSpace(msh, ("Lagrange", 1))
    u = Function(V, dtype=np.float64)
    u.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(2 * x[1] * np.pi))

    # As we want to visualize the function u, we have to create a grid to
    # attached the DoF values to We do this by creating a topology and
    # geometry based on the function space V
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array

    # We set the function "u" as the active scalar for the mesh, and warp
    # the mesh in z-direction by its values
    grid.set_active_scalars("u")
    warped = grid.warp_by_scalar()

    # We create a plotting window consisting of to plots, one of the scalar
    # values, and one where the mesh is warped by these values
    subplotter = pyvista.Plotter(shape=(1, 2))
    subplotter.subplot(0, 0)
    subplotter.add_text("Scalar contour field", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
    subplotter.view_xy()

    subplotter.subplot(0, 1)
    subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
    sargs = dict(height=0.8, width=0.1, vertical=True, position_x=0.05,
                 position_y=0.05, fmt="%1.2e", title_font_size=40, color="black", label_font_size=25)
    subplotter.set_position([-3, 2.6, 0.3])
    subplotter.set_focus([3, -1, -0.15])
    subplotter.set_viewup([0, 0, 1])
    subplotter.add_mesh(warped, show_edges=True, scalar_bar_args=sargs)
    if pyvista.OFF_SCREEN:
        subplotter.screenshot("2D_function_warp.png", transparent_background=transparent,
                              window_size=[figsize, figsize])
    else:
        subplotter.show()

if __name__ == "__main__":
    plot_scalar()