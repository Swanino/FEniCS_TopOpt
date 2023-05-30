import pyvista as pv
import numpy as np
from dolfinx.fem import Function
from dolfinx.cpp.mesh import entities_to_geometry
from mpi4py import MPI

def fenics_to_pyvista_mesh(mesh):
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    cell_entities = np.arange(num_cells, dtype=np.int32)
    
    cell_nodes = entities_to_geometry(mesh, mesh.topology.dim, cell_entities, True)
    num_points = np.max(cell_nodes) + 1
    points = np.zeros([num_points, 3])
    points[:,:mesh.geometry.dim] = mesh.geometry.x
    
    cells = np.hstack((np.full((num_cells, 1), mesh.topology.dim), cell_nodes))
    
    vtk_type = pv.vtk_type(np.dtype('int32'))
    pv_mesh = pv.UnstructuredGrid(cells.ravel(), np.full(num_cells, vtk_type), points)
    
    return pv_mesh

def plot_warped(mesh, function):
    assert isinstance(function, Function)
    
    # Convert FEniCS mesh to PyVista mesh
    pv_mesh = fenics_to_pyvista_mesh(mesh)

    # Get the function values at the mesh nodes
    values = function.compute_point_values()
    
    # Add the function values to the mesh
    pv_mesh["values"] = values
    
    # Warp the mesh by the function values
    warped_mesh = pv_mesh.warp_by_scalar("values")

    # Create a plotter and add the warped mesh
    plotter = pv.Plotter()
    plotter.add_mesh(warped_mesh)
    
    # Display the plot
    plotter.show()