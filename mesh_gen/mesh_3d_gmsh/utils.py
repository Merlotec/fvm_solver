import pyvista as pv
from pyvista.core.pointset import UnstructuredGrid
import numpy as np
import matplotlib.pyplot as plt


def plot_interactive(grid: UnstructuredGrid):
    plotter = pv.Plotter()
    plotter.add_mesh_clip_plane(
        grid,
        normal="x",
        show_edges=True,
    )
    plotter.show_bounds()
    plotter.show()


def plot_slice(mesh: UnstructuredGrid):
    centers = mesh.cell_centers().points
    cell_ids = np.where(centers[:, 0] < 0)[0]

    left_cells = mesh.extract_cells(cell_ids)

    left_cells.plot(show_edges=True)


def plot_clip(grid: UnstructuredGrid):
    grid = grid.clip(normal="z", origin=(0, 0, 0.0))
    grid.plot(show_edges=True, show_axes=True)


def plot_interp(grid: UnstructuredGrid):
    """ Plot interpolated scalar values on a slice plane. Uses nearest-neighbour interpolation from cell centers. """
    from scipy.spatial import cKDTree

    # grid: pyvista.UnstructuredGrid with cell data
    scalar = "volume"

    # Slice plane z = z0
    z0 = grid.center[2]

    nx, ny = 300, 300
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, z0)

    sample_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Cell centers of tetrahedra
    centers = grid.cell_centers().points

    # Nearest-neighbour lookup from sample points to cell centers
    tree = cKDTree(centers)
    dist, idx = tree.query(sample_points)

    values = grid.cell_data[scalar][idx].reshape(ny, nx)

    plt.figure()
    plt.pcolormesh(x, y, values, shading="auto")
    plt.gca().set_aspect("equal")
    plt.colorbar(label=scalar)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Nearest cell value on z = {z0:g}")
    plt.show()
