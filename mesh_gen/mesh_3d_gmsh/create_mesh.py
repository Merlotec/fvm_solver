import numpy as np
import gmsh
import os
import logging
from mesh_gen.mesh_gen_utils import MeshProps

def print_gmsh_quality_metrics():
    # Get all mesh elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    quality_names = [
        "minSICN",     # signed inverted condition number; good general quality metric
        "minSJ",       # scaled Jacobian
        "gamma",       # inscribed/circumscribed radius ratio
        "minEdge",     # smallest edge length
        "maxEdge",     # largest edge length
        "volume",      # area in 2D, volume in 3D
    ]

    for etype, tags in zip(elem_types, elem_tags):
        if etype != 4:
            continue

        if len(tags) == 0:
            continue

        name, dim, order, num_nodes, local_coords, _ = \
            gmsh.model.mesh.getElementProperties(etype)

        print(f"\nElement type {etype}: {name}")
        print(f"Number of elements: {len(tags)}")

        for qname in quality_names:
            try:
                q = np.array(
                    gmsh.model.mesh.getElementQualities(tags, qname),
                    dtype=float
                )

                print(
                    f"  {qname:8s}: "
                    f"min={q.min(): .6e}, "
                    f"mean={q.mean(): .6e}, "
                    f"max={q.max(): .6e}, "
                    f"p01={np.percentile(q, 1): .6e}, "
                    f"p99={np.percentile(q, 99): .6e}"
                )

            except Exception as e:
                print(f"  {qname:8s}: not available ({e})")

def _extract_faces_from_tets(tetra):
    """Extract all unique triangular faces from tetrahedra and classify them.

    Args:
        tetra: (M, 4) integer array of tetrahedra vertex indices.

    Returns:
        int_faces: (K, 3) interior faces (appear in 2 tets).
        bound_faces: (L, 3) boundary faces (appear in 1 tet).
        face_to_tet: None (unused, optimized away for performance).
    """
    if len(tetra) == 0:
        return np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=int), None

    # All 4 faces of each tet, in consistent winding
    face_patterns = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ])

    all_faces_flat = tetra[:, face_patterns].reshape(-1, 3)
    sorted_faces = np.sort(all_faces_flat, axis=1)

    # Fast vectorized deduplication
    unique_faces, counts = np.unique(sorted_faces, axis=0, return_counts=True)

    int_faces = unique_faces[counts == 2]
    bound_faces = unique_faces[counts == 1]

    if np.any(counts > 2):
        logging.warning("Some faces appear in more than 2 tetrahedra!")

    return int_faces, bound_faces, None


def create_mesh_3d(coords: list, mesh_props: MeshProps, quality_kwargs=None):
    """Generate a 3D tetrahedral mesh from a list of MeshSurface3D objects using GMSH.

    Pipeline:
      1. Initialize GMSH and configure mesh size options.
      2. Construct OpenCASCADE geometries and apply boolean cuts for holes.
      3. Set up distance-based refinement fields (if requested).
      4. Generate the 3D mesh and extract nodes and tetrahedra.
      5. Extract interior/boundary faces and assign boundary markers via analytical distance.

    Args:
        coords: list of MeshSurface3D objects defining the domain.
        mesh_props: MeshProps with min_size, max_size, lengthscale.
        quality_kwargs: dictionary of additional options (currently unused, kept for API compatibility).

    Returns:
        mesh_specs: tuple of (
            (points, tetra),          # point (N,3) and cell (M,4) arrays
            (None, face_markers),     # boundary face markers
            (int_faces, bound_faces), # face arrays
        )
        marker_names: dict mapping marker_id → name.
    """
    if quality_kwargs is None:
        quality_kwargs = {}

    gmsh.initialize()
    gmsh.model.add("mesh_3d")

    # Set mesh size bounds
    h_min = 2*mesh_props.min_cell ** (1/3)
    h_max = mesh_props.max_cell ** (1/3)
    print(f'{h_min = }, {h_max = }')
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    
    # Use Algorithm 1 (Delaunay) for 2D and 3D
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    
    # Mesh Optimization (GMSH uses global optimizers instead of simple angle checks)
    gmsh.option.setNumber("Mesh.Optimize", quality_kwargs.get("optimize", 1))
    gmsh.option.setNumber("Mesh.OptimizeNetgen", quality_kwargs.get("optimize_netgen", 1))
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.8)

    domain_tags = []
    hole_tags = []
    dist_req_face_tags = set()

    for i, facet in enumerate(coords):
        dim_tag = facet.build_occ()
        gmsh.model.occ.synchronize()

        bnd_faces = gmsh.model.getBoundary([dim_tag], oriented=False)
        if facet.dist_req:
            dist_req_face_tags.update(f[1] for f in bnd_faces)
        if facet.hole:
            hole_tags.append(dim_tag)
        else:
            domain_tags.append(dim_tag)
            
    # Get bounding boxes for each original facet BEFORE the cut
    facet_bbs = []
    for facet in coords:
        if facet.real_face:
            # We can use the facet's own distance/bounding properties, or just reconstruct temporarily
            dim_tag_tmp = facet.build_occ()
            gmsh.model.occ.synchronize()
            bb = gmsh.model.occ.getBoundingBox(dim_tag_tmp[0], dim_tag_tmp[1])
            gmsh.model.occ.remove([dim_tag_tmp])
            facet_bbs.append(bb)
        else:
            facet_bbs.append(None)

    # Apply boolean cut for holes
    if hole_tags:
        final_vols, _ = gmsh.model.occ.cut(domain_tags, hole_tags)
    else:
        final_vols = domain_tags

    gmsh.model.occ.synchronize()

    # Properly track boundary tags using gmsh bounding boxes to avoid point distance evaluation
    boundary_surfaces = gmsh.model.getBoundary(final_vols, oriented=False)

    boundary_tags = {0: []}
    marker_names = {0: "Normal"}
    for i, facet in enumerate(coords):
        if facet.real_face:
            boundary_tags[i + 1] = []
            marker_names[i + 1] = facet.name

    dist_req_face_tags = []

    eps = 1e-4

    for dim, tag in boundary_surfaces:
        fbb = gmsh.model.getBoundingBox(dim, tag)
        assigned = False
        
        for i, facet in enumerate(coords):
            if not facet.real_face:
                continue
            
            bb = facet_bbs[i]
            # Check if surface bounding box is completely inside the facet bounding box
            if (fbb[0] >= bb[0]-eps and fbb[1] >= bb[1]-eps and fbb[2] >= bb[2]-eps and
                fbb[3] <= bb[3]+eps and fbb[4] <= bb[4]+eps and fbb[5] <= bb[5]+eps):
                
                boundary_tags[i + 1].append(tag)
                if facet.dist_req:
                    dist_req_face_tags.append(tag)
                assigned = True
                break
                
        if not assigned:
            boundary_tags[0].append(tag)

    # Add Physical Groups for properly tracking boundary tags in gmsh
    for mark_id, tags in boundary_tags.items():
        if tags:
            gmsh.model.addPhysicalGroup(2, tags, mark_id)
            gmsh.model.setPhysicalName(2, mark_id, marker_names[mark_id])

    gmsh.model.addPhysicalGroup(3, [v[1] for v in final_vols], 1)
    gmsh.model.setPhysicalName(3, 1, "Domain")

    # Configure mesh refinement fields based on distance to boundaries
    if dist_req_face_tags:
        dist_fields = []
        for face_tag in dist_req_face_tags:
            fid = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fid, "FacesList", [face_tag])
            dist_fields.append(fid)

        min_dist_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_dist_field, "FieldsList", dist_fields)

        final_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(final_field, "InField", min_dist_field)
        gmsh.model.mesh.field.setNumber(final_field, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(final_field, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(final_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(final_field, "DistMax", mesh_props.lengthscale)

        gmsh.model.mesh.field.setAsBackgroundMesh(final_field)

    gmsh.model.mesh.generate(3)
    print_gmsh_quality_metrics()

    # Optimize node extraction using numpy mapping instead of dict comprehension
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    points = np.array(nodeCoords).reshape(-1, 3)

    max_tag = int(np.max(nodeTags)) if len(nodeTags) > 0 else 0
    nodeId_to_idx = np.zeros(max_tag + 1, dtype=np.int64)
    nodeId_to_idx[nodeTags] = np.arange(len(nodeTags))

    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(3)
    tetra_list = [
        nodes.reshape(-1, 4) for etype, nodes in zip(elemTypes, elemNodeTags) if etype == 4
    ]

    if tetra_list:
        tetra_raw = np.vstack(tetra_list)
        tetra = nodeId_to_idx[tetra_raw]
    else:
        tetra = np.zeros((0, 4), dtype=int)

    int_faces, bound_faces, _ = _extract_faces_from_tets(tetra)

    # Extract triangles map from physical groups natively
    face_to_marker = {}
    for pdim, ptg in gmsh.model.getPhysicalGroups(2):
        entities = gmsh.model.getEntitiesForPhysicalGroup(pdim, ptg)
        for ent in entities:
            etypes, etags, enodetags = gmsh.model.mesh.getElements(pdim, ent)
            for eType, nodes in zip(etypes, enodetags):
                if eType == 2:  # Triangle
                    nodes_arr = nodes.reshape(-1, 3)
                    for nd in nodes_arr:
                        tup = tuple(np.sort(nodeId_to_idx[nd]))
                        face_to_marker[tup] = ptg

    # Assign face markers exactly matching the bound_faces arrays
    face_markers = np.ones(len(bound_faces), dtype=int)
    for i, face in enumerate(bound_faces):
        tup = tuple(np.sort(face))
        if tup in face_to_marker:
            face_markers[i] = face_to_marker[tup]

    mesh_specs = (
        (points, tetra),
        (None, face_markers),
        (int_faces, bound_faces),
    )

    gmsh.finalize()
    return mesh_specs, marker_names

