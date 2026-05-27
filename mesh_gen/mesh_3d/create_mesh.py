import numpy as np
import gmsh
import logging
from mesh_gen.mesh_gen_utils import MeshProps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_mesh_options(mesh_props):
    """Configure GMSH mesh sizing and quality options.  Returns (h_min, h_max)."""
    h_min = mesh_props.min_cell ** (1/3)
    h_max = mesh_props.max_cell ** (1/3)
    print(f'{h_min = }, {h_max = }')
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.8)
    return h_min, h_max


def _build_occ(coords):
    """Build OCC volumes and capture their original boundary surfaces.

    Returns:
        domain_info: list of ``(vol_dim_tag, mark_id, dist_req, [face_dim_tags])``
                     for non-hole real_face facets.
        hole_info:   list of ``(vol_dim_tag, mark_id, dist_req, [face_dim_tags])``
                     for hole facets.  *mark_id* is provided for real_face holes
                     (e.g. a "NavierWall" sphere), ``None`` otherwise.
        marker_names: dict ``{mark_id: name}``.
        other_faces:  list of ``(dim, tag)`` for faces of non-real_face facets
                      (e.g. outer domain boundary when you only want a cutout wall).
    """
    domain_info = []
    hole_info = []
    marker_names = {0: "Normal"}
    other_faces = []

    for i, facet in enumerate(coords):
        vol_tag = facet.build_occ()
        gmsh.model.occ.synchronize()
        bnd = gmsh.model.getBoundary([vol_tag], oriented=False)

        mark_id = (i + 1) if facet.real_face else None
        if facet.real_face:
            marker_names[i + 1] = facet.name

        if facet.hole:
            hole_info.append((vol_tag, mark_id, facet.dist_req, bnd))
        else:
            if facet.real_face:
                domain_info.append((vol_tag, mark_id, facet.dist_req, bnd))
            else:
                other_faces.extend(bnd)

    return domain_info, hole_info, marker_names, other_faces


def _fragment_and_remove(domain_info, hole_info, other_faces):
    """Fragment domain + holes, then remove hole-derived volumes.

    Strategy (see test_algo.py):
      1. ``occ.fragment(objects, tools)`` where
           *objects* = all domain surfaces + volumes + other faces,
           *tools*   = all hole surfaces   + volumes.
         ``removeObject=True, removeTool=True`` so originals are cleaned up.
      2. ``out_map[i]`` maps each old entity to its new counterpart(s).
         Use it to find which new surfaces came from which old facet.
      3. ``occ.remove`` only the *hole-derived* new volumes (``recursive=False``)
         so their boundary surfaces survive.
      4. Final volumes = domain-derived minus hole-derived.

    Returns:
        boundary_tags:  ``{mark_id: [surface_tag, …]}``.
        dist_req_tags:  flat list of surface tags needing distance refinement.
        final_vols:     list of ``(dim, tag)`` for remaining 3-D volumes.
    """
    # Build input lists — surfaces FIRST, then volumes (matching test_algo.py)
    objects = []  # domain faces  + volumes  + other_faces
    tools = []    # hole   faces  + volumes

    domain_old_vols = []
    for vol_tag, mark_id, dist_req, faces in domain_info:
        domain_old_vols.append(vol_tag)
        objects.extend(faces)
        objects.append(vol_tag)

    for vol_tag, mark_id, dist_req, faces in hole_info:
        tools.extend(faces)
        tools.append(vol_tag)

    objects.extend(other_faces)

    # --- 1. Fragment ---
    out, out_map = gmsh.model.occ.fragment(objects, tools,
                                           removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    inputs = objects + tools
    # Dict lookup is O(1) and avoids any tuple-equality subtleties with gmsh tags
    idx_of = {old: i for i, old in enumerate(inputs)}

    def _mapped(old_entities):
        result = []
        for old in old_entities:
            result.extend(out_map[idx_of[old]])
        return list(dict.fromkeys(result))  # deduplicate, preserve order

    # --- 2. Remove hole-derived volumes (keep their boundary faces) ---
    hole_vols_new = set()
    for vol_tag, mark_id, _, _ in hole_info:
        for dim, tag in _mapped([vol_tag]):
            if dim == 3:
                hole_vols_new.add((dim, tag))

    if hole_vols_new:
        gmsh.model.occ.remove(list(hole_vols_new), recursive=False)
        gmsh.model.occ.synchronize()

    # --- 3. Collect final volumes (domain-derived minus hole-derived) ---
    domain_vols_new = set()
    for vol_tag in domain_old_vols:
        for dim, tag in _mapped([vol_tag]):
            if dim == 3:
                domain_vols_new.add((dim, tag))
    final_vols = sorted(domain_vols_new - hole_vols_new)

    # --- 4. Match boundary surfaces via out_map ---
    boundary_tags = {}
    dist_req_tags = []

    def _collect_faces(mark_id, dist_req, old_faces):
        new_surf_tags = [tag for dim, tag in _mapped(old_faces) if dim == 2]
        if mark_id is not None:
            boundary_tags[mark_id] = new_surf_tags
        if dist_req:
            dist_req_tags.extend(new_surf_tags)


    for vol_tag, mark_id, dist_req, old_faces in domain_info:
        _collect_faces(mark_id, dist_req, old_faces)

    for vol_tag, mark_id, dist_req, old_faces in hole_info:
        _collect_faces(mark_id, dist_req, old_faces)

    # Unassigned → marker 0
    assigned = set().union(*boundary_tags.values())
    all_bnd = gmsh.model.getBoundary(final_vols, oriented=False)
    boundary_tags[0] = [t for _, t in all_bnd if t not in assigned]

    return boundary_tags, dist_req_tags, final_vols


def _assign_physical_groups(boundary_tags, marker_names, final_vols):
    """Create GMSH physical groups for boundary surfaces and the domain volume."""
    for mark_id, tags in boundary_tags.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, mark_id)
            gmsh.model.setPhysicalName(2, mark_id, marker_names[mark_id])

    gmsh.model.addPhysicalGroup(3, [v[1] for v in final_vols], 1)
    gmsh.model.setPhysicalName(3, 1, "Domain")


def _setup_distance_field(dist_req_tags, h_min, h_max, lengthscale):
    """Create a GMSH background-mesh field for distance-based refinement."""
    if not dist_req_tags:
        return

    dist_fields = []
    for tag in dist_req_tags:
        fid = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fid, "FacesList", [tag])
        dist_fields.append(fid)

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", dist_fields)

    final_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(final_field, "InField", min_field)
    gmsh.model.mesh.field.setNumber(final_field, "SizeMin", h_min)
    gmsh.model.mesh.field.setNumber(final_field, "SizeMax", h_max)
    gmsh.model.mesh.field.setNumber(final_field, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(final_field, "DistMax", lengthscale)
    gmsh.model.mesh.field.setAsBackgroundMesh(final_field)


def _extract_faces_from_tets(tetra):
    """Extract all unique triangular faces from tetrahedra and classify them."""
    if len(tetra) == 0:
        return np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=int), None

    face_patterns = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    all_faces_flat = tetra[:, face_patterns].reshape(-1, 3)
    sorted_faces = np.sort(all_faces_flat, axis=1)
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)

    if np.any(counts > 2):
        logging.warning("Some faces appear in more than 2 tetrahedra!")

    return unique_faces[counts == 2], unique_faces[counts == 1], None


def _extract_mesh_entities():
    """After mesh generation, extract nodes, tetrahedra and faces.

    Returns:
        points:        (N, 3) array.
        nodeId_to_idx: 1-D array mapping gmsh node tag → 0-based index.
        tetra:         (M, 4) array.
        int_faces:     (K, 3) interior faces.
        bound_faces:   (L, 3) boundary faces.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    points = np.array(coords).reshape(-1, 3)

    max_tag = int(np.max(node_tags)) if len(node_tags) > 0 else 0
    nodeId_to_idx = np.zeros(max_tag + 1, dtype=np.int64)
    nodeId_to_idx[node_tags] = np.arange(len(node_tags))

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
    tetra_list = [
        nodes.reshape(-1, 4)
        for etype, nodes in zip(elem_types, elem_nodes) if etype == 4
    ]
    tetra = np.vstack(tetra_list) if tetra_list else np.zeros((0, 4), dtype=int)
    tetra = nodeId_to_idx[tetra]

    int_faces, bound_faces, _ = _extract_faces_from_tets(tetra)
    return points, nodeId_to_idx, tetra, int_faces, bound_faces


def _build_face_markers(bound_faces, nodeId_to_idx):
    """Map each boundary face to its GMSH physical-group marker.

    Returns:
        face_markers: (L,) array of integer marker IDs.
    """
    face_to_marker = {}
    for pdim, ptg in gmsh.model.getPhysicalGroups(2):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(pdim, ptg):
            etypes, _, enodetags = gmsh.model.mesh.getElements(pdim, ent)
            for etype, nodes in zip(etypes, enodetags):
                if etype == 2:                      # triangle
                    for nd in nodes.reshape(-1, 3):
                        face_to_marker[tuple(np.sort(nodeId_to_idx[nd]))] = ptg

    face_markers = np.ones(len(bound_faces), dtype=int)
    for i, face in enumerate(bound_faces):
        face_markers[i] = face_to_marker.get(tuple(np.sort(face)), 1)
    return face_markers


# ---------------------------------------------------------------------------
# Diagnostics (kept as public API)
# ---------------------------------------------------------------------------

def print_gmsh_quality_metrics():
    """Print element-quality statistics for tetrahedral elements."""
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements()
    quality_names = ["minSICN", "minSJ", "gamma", "minEdge", "maxEdge", "volume"]

    for etype, tags in zip(elem_types, elem_tags):
        if etype != 4 or len(tags) == 0:
            continue
        name, *_ = gmsh.model.mesh.getElementProperties(etype)
        print(f"\nElement type {etype}: {name}")
        print(f"Number of elements: {len(tags)}")
        for qn in quality_names:
            try:
                q = np.array(
                    gmsh.model.mesh.getElementQualities(tags, qn), dtype=float,
                )
                print(
                    f"  {qn:8s}: "
                    f"min={q.min():.6e}, "
                    f"mean={q.mean():.6e}, "
                    f"max={q.max():.6e}, "
                    f"p01={np.percentile(q, 1):.6e}, "
                    f"p99={np.percentile(q, 99):.6e}"
                )
            except Exception as e:
                print(f"  {qn:8s}: not available ({e})")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def create_mesh_3d(coords: list, mesh_props: MeshProps):
    """Generate a 3D tetrahedral mesh from a list of ``MeshSurface3D`` objects.

    Pipeline:
      1. Build OCC geometry; fragment domain + holes via ``occ.fragment`` to track
         boundary surfaces; remove hole volumes via ``occ.remove``.
      2. Assign GMSH physical groups.
      3. Set up distance-based refinement fields.
      4. Generate mesh and extract nodes, tetrahedra, faces, and markers.

    Args:
        coords:     list of ``MeshSurface3D`` objects defining the domain.
        mesh_props: ``MeshProps`` with *min_cell*, *max_cell*, *lengthscale*.

    Returns:
        mesh_specs:   ``((points, tetra), (None, face_markers),
                       (int_faces, bound_faces))``
        marker_names: ``{mark_id: name}`` mapping.
    """
    gmsh.initialize()
    gmsh.model.add("mesh_3d")

    # -- 1. Geometry ----------------------------------------------------------
    h_min, h_max = _set_mesh_options(mesh_props)
    domain_info, hole_info, marker_names, other_faces = _build_occ(coords)
    boundary_tags, dist_req_tags, final_vols = _fragment_and_remove(
        domain_info, hole_info, other_faces,
    )
    _assign_physical_groups(boundary_tags, marker_names, final_vols)

    # -- 2. Refinement --------------------------------------------------------
    _setup_distance_field(dist_req_tags, h_min, h_max, mesh_props.lengthscale)

    # -- 3. Mesh generation ---------------------------------------------------
    gmsh.model.mesh.generate(3)
    print_gmsh_quality_metrics()

    # -- 4. Extract data ------------------------------------------------------
    points, nodeId_to_idx, tetra, int_faces, bound_faces = _extract_mesh_entities()
    face_markers = _build_face_markers(bound_faces, nodeId_to_idx)

    gmsh.finalize()
    return (
        (points, tetra),
        (None, face_markers),
        (int_faces, bound_faces),
    ), marker_names
