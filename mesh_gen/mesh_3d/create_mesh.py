import numpy as np
import pyvista as pv
import tetgen
import os
import logging
from scipy.spatial import KDTree

from mesh_gen.mesh_gen_utils import MeshProps


# ---------------------------------------------------------------------------
# Distance-based background-mesh refinement  (3D analog of 2D refine_fn)
# ---------------------------------------------------------------------------


def _build_refinement_bgmesh(merged_surface, dist_req_surfaces, mesh_props):
    """Build a background mesh with per-node ``target_size`` for refinement.

    Mirrors the 2D ``refine_fn`` logic:
      - cells are smallest (``min_cell``) near surfaces tagged ``dist_req=True``,
      - cells grow to ``max_cell`` at distance ≥ ``lengthscale``.

    Uses each surface's analytical ``distance(points)`` method for accuracy.
    tetgen reads the ``"target_size"`` point-data array and uses it as the
    desired local edge-length.
    """
    # 1. Coarse background mesh covering the whole domain
    bg_tet = tetgen.TetGen(merged_surface)
    # bg_tet.add_hole([0, 0, 0])  # add a hole to ensure bgmesh is not a single solid cell
    stdout_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            bg_tet.tetrahedralize(switches="pq1.2a0.2")
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)

    bgmesh = bg_tet.grid
    bgmesh = bgmesh.extract_cells(bgmesh.celltypes == pv.CellType.TETRA).clean()
    bg_points = np.array(bgmesh.points)

    # 2. Minimum signed distance to any dist_req surface, then take absolute value
    dists = [surface.distance(bg_points) for surface in dist_req_surfaces]
    dist = np.min(np.column_stack(dists), axis=1)
    # Clamp negative distances (inside holes) to 0: refinement only needs proximity
    dist = np.maximum(dist, 0)

    # 3. Exponential blend: small cells near surface → large cells far away
    #    (same formula as the 2D refine_fn threshold)
    h_min = mesh_props.min_cell
    h_max = mesh_props.max_cell
    lengthscale = mesh_props.lengthscale

    t = np.clip(dist / lengthscale, 0.0, 1.0)
    sizes = h_min + (h_max - h_min) * t
    bgmesh.point_data["target_size"] = sizes
    return bgmesh


# ---------------------------------------------------------------------------
# Face extraction from tetrahedral grid
# ---------------------------------------------------------------------------
def _extract_faces_from_tets(tetra):
    """Extract all unique triangular faces from tetrahedra and classify them.

    Args:
        tetra: (M, 4) integer array of tetrahedra vertex indices.

    Returns:
        int_faces: (K, 3) interior faces (appear in 2 tets).
        bound_faces: (L, 3) boundary faces (appear in 1 tet).
        face_to_tet: dict mapping sorted face tuple → list of tet indices.
    """
    # All 4 faces of each tet, in consistent winding
    face_patterns = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ])

    all_faces = tetra[:, face_patterns]  # (M, 4, 3)
    all_faces_flat = all_faces.reshape(-1, 3)  # (4M, 3)
    tet_ids = np.repeat(np.arange(len(tetra)), 4)

    # Sort vertices within each face for deduplication
    sorted_faces = np.sort(all_faces_flat, axis=1)

    # Deduplicate: use a dict
    face_dict = {}
    for i in range(len(sorted_faces)):
        key = tuple(sorted_faces[i].tolist())
        if key not in face_dict:
            face_dict[key] = []
        face_dict[key].append(tet_ids[i])

    int_faces = []
    bound_faces = []
    face_to_tet = {}

    for key, tets in face_dict.items():
        face_to_tet[key] = tets
        if len(tets) == 1:
            bound_faces.append(key)
        elif len(tets) == 2:
            int_faces.append(key)
        else:
            logging.warning(f"Face {key} appears in {len(tets)} tets (expected 1 or 2)")

    int_faces = np.array(int_faces, dtype=int) if int_faces else np.zeros((0, 3), dtype=int)
    bound_faces = np.array(bound_faces, dtype=int) if bound_faces else np.zeros((0, 3), dtype=int)

    return int_faces, bound_faces, face_to_tet


def _assign_boundary_markers(points, bound_faces, surface_centers, surface_markers):
    """Assign a marker to each boundary face by nearest-neighbour lookup.

    Args:
        points: (N, 3) vertex coordinates.
        bound_faces: (L, 3) boundary face vertex indices.
        surface_centers: (S, 3) face centers of the input surface mesh.
        surface_markers: (S,) integer markers from the input surface.

    Returns:
        face_markers: (L,) integer markers.
    """
    if len(bound_faces) == 0:
        return np.array([], dtype=int)

    # Compute face centers of boundary faces
    face_verts = points[bound_faces]  # (L, 3, 3)
    face_centers = np.mean(face_verts, axis=1)  # (L, 3)

    tree = KDTree(surface_centers)
    _, idx = tree.query(face_centers)
    return surface_markers[idx]


def _build_switches(mesh_props: MeshProps, quality_kwargs: dict) -> str:
    """Build tetgen switches string from MeshProps and quality kwargs.

    Uses the -a switch for maximum volume constraint (controls cell size).
    """
    max_vol = mesh_props.max_cell * 2.0  # approximate tetra volume from target area

    # Start with quality mesh generation
    switches = "p"

    # Volume constraint
    # switches += f"a0.01"

    # Quality parameters
    if "mindihedral" in quality_kwargs:
        switches += f"q{quality_kwargs["minratio"]:.1f}/{quality_kwargs['mindihedral']:.1f}"

    return switches


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def create_mesh_3d(coords: list, mesh_props: MeshProps, quality_kwargs=None):
    """Generate a 3D tetrahedral mesh from a list of MeshSurface3D objects.

    Analogous to the 2D create_mesh() pipeline:
      1. Assign unique markers to each surface and merge.
      2. Build a background mesh for distance-based refinement (if any dist_req surfaces).
      3. Tetrahedralize with quality/volume switches and optional bgmesh.
      4. Extract points, tetrahedra, interior/boundary faces, and face markers.

    Args:
        coords: list of MeshSurface3D objects defining the domain.
        mesh_props: MeshProps with min_size, max_size, lengthscale.
        quality_kwargs: dict passed to tetgen.tetrahedralize()
                        (e.g. dict(order=1, mindihedral=30, minratio=1.5)).

    Returns:
        mesh_specs: tuple of (
            (points, tetra),          # point and cell arrays
            (None, face_markers),     # markers (point markers not used for 3D)
            (int_faces, bound_faces), # face arrays
        )
        marker_names: dict mapping marker_id → name.
    """
    if quality_kwargs is None:
        quality_kwargs = dict(order=1, mindihedral=20, minratio=1, epsilon=1e-4)
        # quality_kwargs = dict(order=1)

    # ------------------------------------------------------------------
    # 1. Assign markers and merge surfaces
    # ------------------------------------------------------------------
    surfaces, dist_req_surfaces, holes = [], [], []
    marker_names = {0: "Normal"}
    all_surface_centers, all_surface_markers = [], []

    for i, facet in enumerate(coords):
        if facet.real_face:
            mark_id = i + 1
            marker_names[mark_id] = facet.name

            surf = facet.surface.copy()
            # Tag each face with its marker
            n_cells = surf.n_cells
            surf.cell_data["marker"] = np.full(n_cells, mark_id)

            surfaces.append(surf)

            # Collect hole points
            if facet.hole:
                holes.extend(facet.hole)

            # Collect face centers + markers for later boundary marker lookup
            centers = surf.cell_centers().points
            all_surface_centers.append(centers)
            all_surface_markers.append(np.full(len(centers), mark_id))

        # Collect surfaces used for distance-based refinement  (analogous to 2D dist_req)
        if facet.dist_req:
            dist_req_surfaces.append(facet)

    if not surfaces:
        raise ValueError("No real_face surfaces provided.")

    merged_surface = pv.merge(surfaces).clean()

    if all_surface_centers:
        all_surface_centers = np.vstack(all_surface_centers)
        all_surface_markers = np.concatenate(all_surface_markers)
    else:
        all_surface_centers = np.zeros((0, 3))
        all_surface_markers = np.array([], dtype=int)

    # ------------------------------------------------------------------
    # 2. Build background mesh for distance-based refinement (if any dist_req surfaces)
    # ------------------------------------------------------------------
    bgmesh = None
    if dist_req_surfaces:
        bgmesh = _build_refinement_bgmesh(merged_surface, dist_req_surfaces, mesh_props)

    # ------------------------------------------------------------------
    # 3. Build switches and tetrahedralize
    # ------------------------------------------------------------------
    tet = tetgen.TetGen(merged_surface)
    for hole_pt in holes:
        tet.add_hole(hole_pt)

    # Suppress C-level stdout from tetgen
    stdout_fd = os.dup(1)
    try:
        switches = _build_switches(mesh_props, quality_kwargs)
        print(f'{switches = }')
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            # tet.tetrahedralize(switches=switches, bgmesh=bgmesh)
            tet.tetrahedralize(bgmesh=bgmesh, **quality_kwargs)

    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)

    grid = tet.grid

    # Keep only tetrahedral cells
    grid = grid.extract_cells(grid.celltypes == pv.CellType.TETRA).clean()

    # ------------------------------------------------------------------
    # 4. Extract mesh data
    # ------------------------------------------------------------------
    points = np.array(grid.points)
    # Get tetrahedra connectivity: grid.cells is a flat array [4, v0, v1, v2, 3, 4, ...]
    cell_conn = grid.cells.reshape(-1, 5)  # each row: [4, v0, v1, v2, v3]
    tetra = cell_conn[:, 1:].astype(np.int64)

    int_faces, bound_faces, _ = _extract_faces_from_tets(tetra)

    # Assign markers to boundary faces
    face_markers = _assign_boundary_markers(
        points, bound_faces, all_surface_centers, all_surface_markers
    )

    mesh_specs = (
        (points, tetra),
        (None, face_markers),
        (int_faces, bound_faces),
    )

    return mesh_specs, marker_names

