"""Example: configurable 3D mesh generation with tetgen.
Demonstrates the pipeline:
  1. Define domain surfaces (Box3D, Sphere3D, Cylinder3D).
  2. Generate tetrahedral mesh via create_mesh_3d().
  3. Convert to pyvista UnstructuredGrid for visualisation.
"""
import numpy as np
import pyvista as pv
from mesh_gen.mesh_3d.meshes_fvm import gen_mesh_cube_sphere


def build_pyvista_grid(points, tetra):
    """Convert raw (points, tetra) arrays into a pyvista UnstructuredGrid."""
    n_tets = len(tetra)
    cells = np.hstack([np.full((n_tets, 1), 4), tetra]).ravel()
    cell_types = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


def tet_volumes(points, tetra):
    """Compute absolute volume of each tetrahedron. Points (N,3), tetra (M,4)."""
    a = points[tetra[:, 0]]
    b = points[tetra[:, 1]]
    c = points[tetra[:, 2]]
    d = points[tetra[:, 3]]
    return np.abs(np.sum((b - a) * np.cross(c - a, d - a), axis=1)) / 6.0


def cube_sphere_example():
    """Box domain with a spherical hole at the centre."""
    from collections import Counter

    print("=== Cube-with-sphere-hole mesh ===")
    points, tetra, (int_faces, bound_faces), face_tag = gen_mesh_cube_sphere(volume=(0.05, 0.2), cell_lnscale=0.2)
    print(f"  Points: {points.shape[0]}, Tets: {tetra.shape[0]}")
    print(f"  Interior faces: {int_faces.shape[0]}, Boundary faces: {bound_faces.shape[0]}")
    print(f"  Boundary tags: {Counter(face_tag)}")
    grid = build_pyvista_grid(points, tetra)
    grid.cell_data["volume"] = tet_volumes(points, tetra)
    print(f"  Volume range: [{grid.cell_data['volume'].min():.2g}, "
          f"{grid.cell_data['volume'].max():.2g}]")
    return grid


def main():
    from mesh_gen.mesh_3d.utils import plot_interactive, plot_clip, plot_interp

    grid_cube_sphere = cube_sphere_example()

    print("\nOpening interactive viewer (cube + sphere hole) ...")
    # plot_interp(grid_cube_sphere)
    #
    # plot_clip(grid_cube_sphere)
    #
    # plot_interactive(grid_cube_sphere)
#


if __name__ == "__main__":
    main()
