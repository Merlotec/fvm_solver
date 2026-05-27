import numpy as np
from dataclasses import dataclass


@dataclass
class MeshSurface3D:
    """Base class for 3D mesh surfaces used with gmsh.

    Attributes:
        hole: True if this region should be subtracted from the domain.
        dist_req: Whether this surface should influence mesh refinement spacing.
        name: Tag carried through to the mesh.
        real_face: Whether this surface is part of the CFD domain boundary.
    """
    hole: bool = False
    dist_req: bool = True
    name: str | None = None
    real_face: bool = True

    def build_occ(self):
        """Construct the object in gmsh.model.occ and return the (dim, tag) tuple."""
        raise NotImplementedError

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points (N,3) to this surface.
        Override in subclasses.
        """
        raise NotImplementedError


class Box3D(MeshSurface3D):
    def __init__(self, Xmin, Xmax, hole: bool = False, dist_req: bool = False, name=None, real_face=True):
        super().__init__(hole=hole, dist_req=dist_req, name=name, real_face=real_face)
        self.xmin = np.asarray(Xmin, dtype=float)
        self.xmax = np.asarray(Xmax, dtype=float)

    def build_occ(self):
        import gmsh
        dx, dy, dz = self.xmax - self.xmin
        tag = gmsh.model.occ.addBox(self.xmin[0], self.xmin[1], self.xmin[2], dx, dy, dz)
        return (3, tag)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the box surface."""
        _box_center = (self.xmin + self.xmax) / 2
        _half_extent = (self.xmax - self.xmin) / 2
        p = np.atleast_2d(points) - _box_center
        q = np.abs(p) - _half_extent
        d_outer = np.linalg.norm(np.maximum(q, 0), axis=1)
        d_inner = np.minimum(np.max(q, axis=1), 0)
        return d_outer + d_inner


class Sphere3D(MeshSurface3D):
    def __init__(self, center, radius,
                 hole: bool = True, dist_req: bool = True, name=None):
        super().__init__(hole=hole, dist_req=dist_req, name=name, real_face=True)
        self._center = np.asarray(center, dtype=float)
        self._radius = float(radius)

    def build_occ(self):
        import gmsh
        tag = gmsh.model.occ.addSphere(self._center[0], self._center[1], self._center[2], self._radius)
        return (3, tag)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the sphere surface."""
        return np.linalg.norm(np.atleast_2d(points) - self._center, axis=1) - self._radius


class Cylinder3D(MeshSurface3D):
    def __init__(self, center, radius, height, resolution=32,
                 hole: bool = True, dist_req: bool = True, name=None):
        super().__init__(hole=hole, dist_req=dist_req, name=name, real_face=True)
        self._center = np.asarray(center, dtype=float)
        self._radius = float(radius)
        self._height = float(height)

    def build_occ(self):
        import gmsh
        # gmsh cylinder is specified by center of one base, and vector to other base
        # Center is the midpoint in geometries.py, so go from center - height/2 to center + height/2
        z0 = self._center[2] - self._height / 2
        dz = self._height
        tag = gmsh.model.occ.addCylinder(self._center[0], self._center[1], z0, 0, 0, dz, self._radius)
        return (3, tag)

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from query points to the capped cylinder surface."""
        self._half_height = self._height / 2
        p = np.atleast_2d(points) - self._center
        rho = np.linalg.norm(p[:, :2], axis=1)
        z = np.abs(p[:, 2])

        d_cyl = rho - self._radius
        d_cap = z - self._half_height

        d_outside_both = np.sqrt(np.maximum(d_cyl, 0)**2 + np.maximum(d_cap, 0)**2)
        d_inside = np.minimum(np.maximum(d_cyl, d_cap), 0)

        return d_outside_both + d_inside

