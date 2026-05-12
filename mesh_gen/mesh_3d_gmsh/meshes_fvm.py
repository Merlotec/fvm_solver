import numpy as np

from mesh_gen.mesh_gen_utils import MeshProps
from .geometries import Box3D, Sphere3D, Cylinder3D
from .create_mesh import create_mesh_3d


def gen_mesh_cube_sphere(volume, cell_lnscale=2.0):
    """Generate a 3D mesh: unit cube with a spherical hole at the center.

    Args:
        volume: (min_vol, max_vol) tuple controlling cell sizes.
        cell_lnscale: lengthscale controlling refinement transition zone width.

    Returns:
        points: (N, 3) vertex coordinates.
        tetra: (M, 4) tetrahedra vertex indices.
        faces: (int_faces, bound_faces) tuple of face index arrays.
        face_tag: list of boundary face name strings.
    """
    L = 2.0
    hole_radius = 0.45
    hole_center = np.array([0.0, 0.0, 0.0])

    min_vol, max_vol = volume
    mesh_lnscale = min_vol**(1/3)

    mesh_props = MeshProps(min_vol, max_vol, lengthscale=cell_lnscale)

    coords = [
        Box3D(
            Xmin=[-L/2, -L/2, -L/2],
            Xmax=[L/2, L/2, L/2],
            hole=False,
            dist_req=False,
            name="Farfield",
        ),
        Sphere3D(
            center=hole_center, radius=hole_radius,
            hole=True, dist_req=True, 
            name="NavierWall",
        ),
    ]

    mesh_specs, marker_tags = create_mesh_3d(coords, mesh_props)

    _point_props, _markers, _faces = mesh_specs
    points, tetra = _point_props
    _, face_markers = _markers
    int_faces, bound_faces = _faces

    # Map marker IDs back to names
    face_tag = [marker_tags[int(i)] for i in face_markers]

    return points, tetra, (int_faces, bound_faces), face_tag


def gen_mesh_pipe(areas, cell_lnscale=2.0):
    """Generate a 3D mesh: box with a cylindrical hole along the z-axis.

    The cylinder is fully contained within the box domain.

    Args:
        areas: (min_area, max_area) tuple controlling cell sizes.
        cell_lnscale: lengthscale controlling refinement transition zone width.

    Returns:
        points: (N, 3) vertex coordinates.
        tetra: (M, 4) tetrahedra vertex indices.
        faces: (int_faces, bound_faces) tuple of face index arrays.
        face_tag: list of boundary face name strings.
    """
    L = 4.0
    pipe_radius = 0.5
    pipe_height = L * 0.8  # fully contained within the box
    pipe_center = np.array([0.0, 0.0, 0.0])

    min_area, max_area = areas
    mesh_props = MeshProps(min_area, max_area, lengthscale=cell_lnscale)

    coords = [
        Box3D(
            Xmin=[-L/2, -L/2, -L/2],
            Xmax=[L/2, L/2, L/2],
            hole=False,
            dist_req=False,
            name="Farfield",
        ),
        Cylinder3D(
            center=pipe_center,
            radius=pipe_radius,
            height=pipe_height,
            hole=True,
            dist_req=True,
            name="NavierWall",
        ),
    ]

    mesh_specs, marker_tags = create_mesh_3d(coords, mesh_props)

    _point_props, _markers, _faces = mesh_specs
    points, tetra = _point_props
    _, face_markers = _markers
    int_faces, bound_faces = _faces

    face_tag = [marker_tags[int(i)] for i in face_markers]

    return points, tetra, (int_faces, bound_faces), face_tag
