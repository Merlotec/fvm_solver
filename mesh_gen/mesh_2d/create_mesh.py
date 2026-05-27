"""2D mesh generation using gmsh (replaces meshpy/Triangle which segfaults on some platforms)."""
from __future__ import annotations
import numpy as np
import logging


class MeshGenerationError(RuntimeError):
    pass


def extract_interor_edges(triangles):
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])
    all_edges = np.sort(edges, axis=1)
    unique_all_edges, all_counts = np.unique(all_edges, axis=0, return_counts=True)
    return unique_all_edges[all_counts == 2]


def _order_segments(seg_list):
    """Order (pt_a, pt_b, line_tag, mark_id) tuples into a connected closed loop.

    Returns list of (pt_a, pt_b, line_tag, mark_id, forward) where
    forward=False means the line should be traversed in reverse (-tag).
    """
    if not seg_list:
        return []

    remaining = list(range(len(seg_list)))
    first = seg_list[0]
    ordered = [(*first, True)]
    remaining.remove(0)
    cur_end = first[1]

    while remaining:
        found = False
        for i in remaining:
            a, b, lt, mk = seg_list[i]
            if a == cur_end:
                ordered.append((a, b, lt, mk, True))
                cur_end = b
                remaining.remove(i)
                found = True
                break
            elif b == cur_end:
                ordered.append((b, a, lt, mk, False))
                cur_end = a
                remaining.remove(i)
                found = True
                break
        if not found:
            logging.warning("_order_segments: could not close loop, %d segment(s) disconnected.", len(remaining))
            break

    return ordered


def _gmsh_create_mesh(coords, mesh_props, min_angle):
    """Build a 2D triangular mesh from a list of MeshFacet objects using gmsh."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        # Disable automatic sizing from geometry — we use a field instead.
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.model.add("mesh2d")

        # Mesh size targets (convert area → approximate edge length)
        min_size = float(np.sqrt(2.0 * mesh_props.min_cell))
        max_size = float(np.sqrt(2.0 * mesh_props.max_cell))
        L = float(mesh_props.lengthscale)

        # ---- point registry (merge coincident vertices) ----
        _pt_map: dict[tuple, int] = {}
        _pt_coords: dict[int, tuple[float, float]] = {}

        def get_pt(x: float, y: float) -> int:
            key = (round(float(x), 8), round(float(y), 8))
            if key not in _pt_map:
                tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0)
                _pt_map[key] = tag
                _pt_coords[tag] = key
            return _pt_map[key]

        # ---- classify facets and build gmsh geometry ----
        refinement_curve_tags: list[int] = []
        outer_segs: list[tuple] = []   # segments forming the outer boundary
        hole_loops: list[int] = []      # curve loop tags for holes
        mark_to_curves: dict[int, list[int]] = {}

        for facet_idx, facet in enumerate(coords):
            mark_id = facet_idx + 1

            if not facet.real_face:
                # Refinement-only: add curves for the distance size field only.
                if facet.dist_req:
                    pt_tags = [get_pt(p[0], p[1]) for p in facet.points]
                    for seg in facet.segments:
                        lt = gmsh.model.geo.addLine(pt_tags[int(seg[0])], pt_tags[int(seg[1])])
                        refinement_curve_tags.append(lt)
                continue

            pt_tags = [get_pt(p[0], p[1]) for p in facet.points]
            seg_info: list[tuple] = []
            for seg in facet.segments:
                a_tag = pt_tags[int(seg[0])]
                b_tag = pt_tags[int(seg[1])]
                lt = gmsh.model.geo.addLine(a_tag, b_tag)
                seg_info.append((a_tag, b_tag, lt, mark_id))
                if facet.dist_req:
                    refinement_curve_tags.append(lt)
                mark_to_curves.setdefault(mark_id, []).append(lt)

            if facet.hole:
                # Closed polygon → form a curve loop.
                # The MeshFacet polygons are parameterised CCW; holes need CW in gmsh,
                # so reverse and negate the curve tags.
                ordered = _order_segments(seg_info)
                ccw_tags = [lt if fwd else -lt for _, _, lt, _, fwd in ordered]
                cw_tags = [-ct for ct in reversed(ccw_tags)]
                cl = gmsh.model.geo.addCurveLoop(cw_tags)
                hole_loops.append(cl)
            else:
                outer_segs.extend(seg_info)

        # ---- outer boundary ----
        ordered_outer = _order_segments(outer_segs)
        outer_curve_tags = [lt if fwd else -lt for _, _, lt, _, fwd in ordered_outer]
        outer_loop = gmsh.model.geo.addCurveLoop(outer_curve_tags)

        # ---- plane surface: outer minus holes ----
        surface = gmsh.model.geo.addPlaneSurface([outer_loop] + hole_loops)

        gmsh.model.geo.synchronize()

        # ---- physical groups (boundary markers) ----
        for mark_id, curve_tags in mark_to_curves.items():
            gmsh.model.addPhysicalGroup(1, curve_tags, mark_id)
        gmsh.model.addPhysicalGroup(2, [surface], 1)

        # ---- size field: distance-based refinement ----
        unique_ref_tags = list(set(refinement_curve_tags))
        if unique_ref_tags:
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", unique_ref_tags)
            gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)

            fm = gmsh.model.mesh.field.add("MathEval")
            # size = (max - min) * (1 - exp(-dist / L)) + min
            gmsh.model.mesh.field.setString(
                fm, "F",
                f"({max_size} - {min_size}) * (1 - Exp(-F{fd} / {L})) + {min_size}"
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(fm)
        else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

        # ---- mesh algorithm ----
        # Algorithm 6 = Frontal-Delaunay (good quality, respects size fields)
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")

        # ---- extract nodes ----
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        pts = node_coords.reshape(-1, 3)[:, :2].astype(np.float64)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # ---- extract triangles ----
        elem_types, _, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)
        triangles = None
        for et, entags in zip(elem_types, elem_node_tags_list):
            if int(et) == 2:  # 3-node triangle
                raw = entags.reshape(-1, 3)
                triangles = np.array(
                    [[tag_to_idx[int(t)] for t in row] for row in raw], dtype=np.int32
                )
                break

        if triangles is None or len(triangles) == 0:
            raise MeshGenerationError("gmsh produced no triangles — check geometry.")

        # ---- extract boundary edges with markers ----
        bound_edges_list: list[list[int]] = []
        f_markers_list: list[int] = []
        for mark_id in mark_to_curves:
            entities = gmsh.model.getEntitiesForPhysicalGroup(1, mark_id)
            for ent in entities:
                et_list, _, en_list = gmsh.model.mesh.getElements(dim=1, tag=ent)
                for et, entags in zip(et_list, en_list):
                    if int(et) == 1:
                        for edge in entags.reshape(-1, 2):
                            bound_edges_list.append([tag_to_idx[int(edge[0])], tag_to_idx[int(edge[1])]])
                            f_markers_list.append(mark_id)

        bound_edges = (np.array(bound_edges_list, dtype=np.int32)
                       if bound_edges_list else np.zeros((0, 2), dtype=np.int32))
        f_markers = np.array(f_markers_list, dtype=np.int32)
        int_edges = extract_interor_edges(triangles)

        # Simple point markers: 0 = interior, mark_id = boundary
        p_markers = np.zeros(len(pts), dtype=np.int32)
        for (a, b), mk in zip(bound_edges_list, f_markers_list):
            p_markers[a] = mk
            p_markers[b] = mk

        return (pts, triangles), (p_markers, f_markers), (int_edges, bound_edges)

    finally:
        gmsh.finalize()


def create_mesh(coords: list, mesh_props, min_angle=None):
    """Generate a 2D triangular mesh from MeshFacet objects.

    Returns (mesh_specs, marker_names) with the same format as before:
        mesh_specs = ((points, triangles), (p_markers, f_markers), (int_edges, bound_edges))
        marker_names = {mark_id: name_str, ...}
    """
    marker_names = {0: "Normal"}
    for i, facet in enumerate(coords):
        if facet.real_face and facet.name is not None:
            marker_names[i + 1] = facet.name

    try:
        mesh_specs = _gmsh_create_mesh(coords, mesh_props, min_angle)
    except MeshGenerationError:
        raise
    except Exception as e:
        raise MeshGenerationError(f"gmsh mesh generation failed: {e}") from e

    return mesh_specs, marker_names
