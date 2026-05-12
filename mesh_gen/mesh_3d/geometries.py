import numpy as np
import pyvista as pv
from dataclasses import dataclass
from scipy.spatial import KDTree


@dataclass
class MeshSurface3D:
    """Base class for 3D mesh surfaces used with tetgen.

    Attributes:
        surface: pyvista PolyData surface mesh (triangulated).
        hole: False if solid region, otherwise list of 3D points inside each hole cavity.
        dist_req: Whether this surface should influence mesh refinement spacing.
        name: Tag carried through to the mesh (e.g. "NavierWall", "Inlet").
        real_face: Whether this surface is part of the CFD domain boundary.
                   Set False for surfaces used only for refinement control.
    """
    surface: pv.PolyData
    hole: bool | list = False
    dist_req: bool = True
    name: str | None = None
    real_face: bool = True

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query *points* (N,3) to this surface.

        Positive = outside / away from the surface.
        Default: Euclidean distance to the nearest surface vertex (KDTree).
        Override in subclasses for analytical formulas.
        """
        tree = KDTree(np.array(self.surface.points))
        dist, _ = tree.query(np.atleast_2d(points))
        return np.atleast_1d(np.asarray(dist, dtype=float))


class Box3D(MeshSurface3D):
    """Axis-aligned box domain defined by min/max corners."""

    def __init__(self, Xmin, Xmax, hole: bool = False, dist_req: bool = False, name=None, real_face=True):
        center = tuple((np.array(Xmin) + np.array(Xmax)) / 2)
        lengths = tuple(np.array(Xmax) - np.array(Xmin))

        surface = pv.Cube(
            center=center,
            x_length=lengths[0],
            y_length=lengths[1],
            z_length=lengths[2],
        ).triangulate()

        if hole:
            hole_pts = [list(center)]
        else:
            hole_pts = False

        super().__init__(
            surface=surface,
            hole=hole_pts,
            dist_req=dist_req,
            name=name,
            real_face=real_face,
        )

        # Store for analytical distance
        self.xmin = np.asarray(Xmin, dtype=float)
        self.xmax = np.asarray(Xmax, dtype=float)
        self._half_extent = (self.xmax - self.xmin) / 2
        self._box_center = (self.xmin + self.xmax) / 2

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the box surface.

        Positive outside the box, negative inside.
        """
        p = np.atleast_2d(points) - self._box_center
        q = np.abs(p) - self._half_extent
        # Distance for points outside the box (clamp negative components to 0)
        d_outer = np.linalg.norm(np.maximum(q, 0), axis=1)
        # Distance for points inside the box (max of the most-negative q component)
        d_inner = np.minimum(np.max(q, axis=1), 0)
        # Signed distance: positive outside, negative inside
        return d_outer + d_inner


class Sphere3D(MeshSurface3D):
    """Sphere surface, typically used as a hole in the domain."""

    def __init__(self, center, radius, mesh_lnscale=None, theta_resolution=16, phi_resolution=16,
                 hole: bool = True, dist_req: bool = True, name=None):
        if mesh_lnscale is not None:
            circumference = np.pi * radius
            theta_resolution = int(np.ceil(mesh_lnscale * theta_resolution / circumference))
            phi_resolution = int(np.ceil(mesh_lnscale * phi_resolution / circumference))

        surface = pv.Sphere(
            center=tuple(center), radius=radius,
            theta_resolution=theta_resolution, phi_resolution=phi_resolution,
        ).triangulate()

        if hole:
            hole_pts = [list(center)]
        else:
            hole_pts = False

        super().__init__(
            surface=surface,
            hole=hole_pts,
            dist_req=dist_req,
            name=name,
            real_face=True,
        )

        self._center = np.asarray(center, dtype=float)
        self._radius = float(radius)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the sphere surface.

        Positive outside the sphere, negative inside.
        """
        return np.linalg.norm(np.atleast_2d(points) - self._center, axis=1) - self._radius


class Cylinder3D(MeshSurface3D):
    """Cylinder surface along the z-axis, typically used as a hole.

    Uses pyvista's capped cylinder and triangulates it for tetgen.
    """

    def __init__(self, center, radius, height, resolution=32,
                 hole: bool = True, dist_req: bool = True, name=None):
        surface = pv.Cylinder(
            center=tuple(center),
            radius=radius,
            height=height,
            resolution=resolution,
            capping=True,
        ).triangulate()

        if hole:
            hole_pts = [list(center)]
        else:
            hole_pts = False

        super().__init__(
            surface=surface,
            hole=hole_pts,
            dist_req=dist_req,
            name=name,
            real_face=True,
        )

        self._center = np.asarray(center, dtype=float)
        self._radius = float(radius)
        self._half_height = float(height) / 2

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the capped cylinder surface.

        The cylinder axis runs along z, centered at _center with radius and half_height.
        Positive outside the cylinder, negative inside.
        """
        p = np.atleast_2d(points) - self._center
        # Radial distance in xy-plane
        rho = np.linalg.norm(p[:, :2], axis=1)
        # Axial distance (absolute z)
        z = np.abs(p[:, 2])

        # Distance to the cylindrical wall (infinite cylinder)
        d_cyl = rho - self._radius

        # Distance to the top/bottom caps (infinite slab of thickness 2*half_height)
        d_cap = z - self._half_height

        # Three regions:
        # 1. Outside both cap and cylinder → Euclidean distance to the closest edge circle
        # 2. Outside cylinder, between caps → radial distance
        # 3. Outside caps, inside cylinder → axial distance
        # 4. Inside both → negative distance to closest surface

        # When both d_cyl > 0 and d_cap > 0: use 2-norm of the two distances
        d_outside_both = np.sqrt(np.maximum(d_cyl, 0)**2 + np.maximum(d_cap, 0)**2)
        # When inside both: max of the two negative distances (closest to a surface)
        d_inside = np.minimum(np.maximum(d_cyl, d_cap), 0)

        return d_outside_both + d_inside
