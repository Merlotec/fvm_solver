import torch
from cprint import c_print


def build_sparse_gradient_matrix(combined_neigh, G_mat, dim, n_cells, n_boundaries):
    """
    Build a sparse gradient matrix for one spatial dimension using both cell neighbors and boundary facets.

    Args:
        combined_neigh Tensor: For each cell i, a 1D tensor of neighbor cell indices.
        G_mat (Tensor): For each cell i, a 2D tensor of shape [n_dims, num_total_neighbors_i].
                              The first columns correspond to cell neighbors and the remaining columns to facets.
        dim (int): The spatial dimension (0 for x, 1 for y) to build the gradient matrix.
        n_cells (int): Total number of cells.
        n_boundaries (int): Total number of boundary facets.

    Returns:
        A (torch.sparse.FloatTensor): A sparse matrix of shape [n_cells, n_cells+n_boundaries] that computes
                                      the gradient along the given dimension.
    """
    rows, cols, vals = [], [], []
    for i in range(n_cells):
        diag_val = 0.0
        # Loop over the combined neighbors.
        for k in range(combined_neigh[i].shape[0]):
            neighbor_idx = int(combined_neigh[i, k].item())
            # Use the appropriate weight from G_mat[i] (first cell_neigh.shape[0] entries correspond to cells)
            g_val = G_mat[i][dim, k]
            rows.append(i)
            cols.append(neighbor_idx)
            vals.append(g_val)
            diag_val += g_val

        # Diagonal entry: subtract the sum of all off-diagonals.
        rows.append(i)
        cols.append(i)
        vals.append(-diag_val)

    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=G_mat[0].dtype)
    A = torch.sparse_coo_tensor(indices, values, (n_cells, n_cells + n_boundaries))
    return A


class FVMMesh:
    # Local only for saving / plotting
    facets: torch.Tensor  # shape = (n_facet, n_nodes_per_facet)
    vertices: torch.Tensor  # shape = (n_vertices, n_dims)
    cells: torch.Tensor  # shape = (n_cells, n_nodes_per_cell)

    # Used for FVM calculations
    bc_facet_mask: torch.Tensor  # shape = (n_facet)
    cell_to_facet: torch.Tensor  # shape = (n_cells, n_facets_per_cell)
    cell_facet_signs: torch.Tensor  # shape = (n_cells, n_facets_per_cell)

    # Only for interior facet
    normals_main: torch.Tensor  # shape = (n_facet_main, n_dims)
    cell_grad_stuff: tuple # Stuff needed to calculate gradient on a cell
    facet_to_cell_main: torch.Tensor # shape = (n_facet_main, 2)              # Mapping facet to cell indices for non-boundary facet

    def _cell_facet_sign(self, centroids, midpoints, cell_to_facet, normals, facet_to_cell, cell_facet_idxs):
        """ Compute which cell is on the left and right of each facet.
            For ordering, cell on Left comes first, then right
            Signs: 1 if on left, -1 if on right.
            Works in any dimension (2D or 3D).
        """
        midpoints_cell = midpoints[cell_to_facet]  # shape: [n_cells, n_facets_per_cell, n_dims]
        normals_cell = normals[cell_to_facet]  # shape: [n_cells, n_facets_per_cell, n_dims]
        # Compute the difference between each facet midpoint and the centroid.
        p_diff = midpoints_cell - centroids.unsqueeze(1)  # shape: [n_cells, n_facets_per_cell, n_dims]
        # Normalize the normals along the last dimension.
        norms = torch.norm(normals_cell, dim=-1, keepdim=True)  # shape: [n_cells, n_facets_per_cell, 1]
        norm_hat = normals_cell / norms  # shape: [n_cells, n_facets_per_cell, n_dims]
        # Compute the dot product and then its sign.
        dist_dot = torch.sum(norm_hat * p_diff, dim=-1)  # shape: [n_cells, n_facets_per_cell]
        signs = torch.sign(dist_dot).long()  # shape: [n_cells, n_facets_per_cell]
        # The displacement vectors are simply p_diff.
        cent_to_facet_disp = p_diff  # shape: [n_cells, n_facets_per_cell, n_dims]

        facet_to_cell_ordered = {}
        p_m, m_p = torch.tensor([1, -1]), torch.tensor([-1, 1])
        for facet in sorted(facet_to_cell.keys()):
            cell_idx = facet_to_cell[facet]
            cell_facet = cell_facet_idxs[facet]

            order = signs[cell_idx, cell_facet]

            # Boundary facets only have 1 cell
            if order.shape[0] == 1:
                assert self.bc_facet_mask[facet] == True, "Inconsistent boundary bug"
            else:
                if torch.all(order == m_p):
                    cell_idx = torch.flip(cell_idx, dims=[0])
            facet_to_cell_ordered[facet] = cell_idx

        return signs, facet_to_cell_ordered, cent_to_facet_disp


class FVMMesh2D(FVMMesh):
    n_cells: int
    n_facets: int
    n_bc_facet: int

    # Used for FVM calculations
    bc_facet_mask: torch.Tensor  # shape = (n_facet)
    areas: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_facet, 2)
    lengths: torch.Tensor  # shape = (n_facet)
    centroids: torch.Tensor  # shape = (n_cells, 2)
    midpoints: torch.Tensor  # shape = (n_facet, 2)

    def __init__(self, vertices, cells, facets, bc_facet_mask, device="cuda"):
        super().__init__()
        self.vertices = vertices
        self.cells = cells
        self.facets = facets
        self.bc_facet_mask = bc_facet_mask
        self.device = device

        self.n_cells = cells.shape[0]
        self.n_facets = facets.shape[0]
        self.n_bc_facet = bc_facet_mask.sum().item()
        assert facets.shape[0] == bc_facet_mask.shape[0], f'Different number of facets from bc facet mask {facets.shape = }, {bc_facet_mask.shape = }'

        c_print(f'Computing mesh properties', color="bright_magenta")
        self._compute_facet_props(vertices, cells, facets)

    def _compute_facet_props(self, vertices, cells, facets):
        print("Compute facet normals and lengths")
        # Compute facet normals and lengths
        facet_vertex = vertices[facets]
        facet_vectors = facet_vertex[:, 1] - facet_vertex[:, 0]        # Ordering is used as facet index from here.
        normals = torch.stack([facet_vectors[:, 1], -facet_vectors[:, 0]], dim=1)
        midpoints = torch.mean(facet_vertex, dim=1)      # shape = [n_facet, 2]
        self.facet_vertex = facet_vertex                  # shape = [n_facet, 2, 2]
        self.normals = normals                          # shape = [n_facet, 2]
        self.midpoints = midpoints                      # shape = [n_facet, 2]

        # Cell area and centroid
        cell_points = vertices[cells]
        self.areas = self._cell_area(cell_points)
        self.centroids = torch.mean(cell_points, dim=1)  # shape = [n_cells, 2]

        # Compute mapping of facet_to_cells and cells_to_facet
        cell_to_facet, _facet_to_cell, cell_facet_idxs = self._get_cell_facets(cells, facets) # shape = [n_cells, 3]
        self.cell_to_facet = cell_to_facet

        # Sort cell in order of facet signed direction. ORDER: [-, +], so cell on right comes first.
        self.cell_facet_signs, facet_to_cell, cent_to_facet_disp = self._cell_facet_sign(self.centroids, midpoints, cell_to_facet, self.normals, _facet_to_cell, cell_facet_idxs)

        # Split tensors into facet and main
        normals_main, facet_to_cell_main = [], []
        facet_to_cell_bc, normals_bc = [], []
        for e_idx, e_bc in enumerate(self.bc_facet_mask):
            if e_bc:
                # Precompute tensors for boundary facets
                facet_to_cell_bc.append(facet_to_cell[e_idx])
                normals_bc.append(normals[e_idx])
            else:
                # Precompute tensors for interior facets
                normals_main.append(normals[e_idx])
                facet_to_cell_main.append(facet_to_cell[e_idx])

        self.normals_main = torch.stack(normals_main)
        self.facet_to_cell_main = torch.stack(facet_to_cell_main)
        self.cent_to_facet_disp = cent_to_facet_disp
        self.facet_to_cell_bc = torch.stack(facet_to_cell_bc).squeeze()

        print(f'Compute grad weighting')
        # Compute grad weighting
        self.cell_grad_stuff = self._grad_weighting(cell_to_facet, facet_to_cell, self.centroids, midpoints, normals)

        print("Done")

    def _cell_area(self, vertices):
        """ vertices.shape = (n_cells, 3, 2) """
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        # Compute the vectors for each cell
        ab = b - a  # shape [n, 2]
        ac = c - a  # shape [n, 2]
        # Compute the 2D cross product (determinant) for each cell
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # shape [n]
        # Cell area is half the absolute value of the cross product
        area = 0.5 * torch.abs(cross)

        return area

    def _get_cell_facets(self, cells, facets):
        """
            Compute which facets belong to each cell, and vice versa
            cells.shape = (n_cells, 3)
            facets.shape = (n_facet, 2)
        """
        # 1) Normalize each facet (sort nodes in ascending order).
        # -------------------------------------------------------
        # facets_sorted will be shape [m, 2] with each row sorted.
        facets_sorted, _ = facets.sort(dim=1)

        # 2) Build a lookup: (nodeA, nodeB) -> facet_index
        # -----------------------------------------------
        facet_dict = {}
        for idx, e in enumerate(facets_sorted):
            # Make a tuple key (nodeA, nodeB)
            key = (e[0].item(), e[1].item())
            facet_dict[key] = idx

        # 3) For each cell, find the 3 facets
        # --------------------------------------
        # We'll create a result tensor of shape [num_cells, 3],
        # each row will store the indices of the 3 facets of that cell.

        cell_to_facet = []
        for cell in cells:
            # Extract cell nodes (v0, v1, v2)
            v0 = cell[0].item()
            v1 = cell[1].item()
            v2 = cell[2].item()

            # Sort each pair so we can look it up in the facet_dict
            e1 = tuple(sorted((v0, v1)))
            e2 = tuple(sorted((v1, v2)))
            e3 = tuple(sorted((v2, v0)))

            # Get the facet indices
            facet_indices = [facet_dict[e1], facet_dict[e2], facet_dict[e3]]
            cell_to_facet.append(facet_indices)

        cell_to_facet = torch.tensor(cell_to_facet)

        # Compute facet to cell
        """ Flattens the cell_to_facet tensor, computes the corresponding cell and local facet indices once, 
        then sorts the flattened entries by facet ID so equal facets become contiguous. 
        The grouped sorted arrays are then split by facet counts, giving all cells and local facet positions 
        for each unique facet without repeatedly scanning the full tensor."""
        unique_facets, inverse = torch.unique(cell_to_facet.reshape(-1), sorted=True, return_inverse=True, )

        num_cells, facets_per_cell = self.n_cells, 3
        flat_pos = torch.arange(cell_to_facet.numel(), device=cell_to_facet.device)

        cell_ids = flat_pos // facets_per_cell
        local_facet_ids = flat_pos % facets_per_cell

        order = torch.argsort(inverse)

        sorted_inverse = inverse[order]
        sorted_cells = cell_ids[order]
        sorted_local_facets = local_facet_ids[order]

        counts = torch.bincount(sorted_inverse, minlength=len(unique_facets))

        cell_groups = torch.split(sorted_cells, counts.tolist())
        local_facet_groups = torch.split(sorted_local_facets, counts.tolist())

        facet_to_cell = {
            int(f.item()): cells
            for f, cells in zip(unique_facets, cell_groups)
        }

        cell_facet_idxs = {
            int(f.item()): idxs
            for f, idxs in zip(unique_facets, local_facet_groups)
        }

        return cell_to_facet, facet_to_cell, cell_facet_idxs

    def _grad_weighting(self, cell_to_facet, facet_to_cell_ord, centroids, midpoints, normals):
        """ Use least squares formula to compute gradient weighting.
            grad(u) = A^-1 * b
            A = sum_i (d_i d_i^T)
            b = sum_i d_i (u_i - u_c)
        """
        bound_facet_idxs = torch.nonzero(self.bc_facet_mask, as_tuple=False).flatten()
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(bound_facet_idxs)}

        combined_neigh, neigh_cents, combined_bc = [], [], []
        for cell_id, facets in enumerate(cell_to_facet):  # Must keep this order. Neighbor id: torch.cat([Us, Us_bc_facet])
            # Get neighboring cells
            neighbors, centers, is_bc = [], [], []
            for e in facets:
                e = e.item()
                if len(facet_to_cell_ord[e]) == 2:
                    """ Interior facet"""
                    is_bc.append(False)
                    cells = facet_to_cell_ord[e]
                    neigh_cell = cells[cells != cell_id]
                    centers.append(centroids[neigh_cell])  # [1, 2]
                    neighbors.append(neigh_cell.item())
                else:
                    """ Boundary facet """
                    is_bc.append(True)
                    midpoint = midpoints[e].unsqueeze(0)
                    centers.append(midpoint)
                    glob_facet_idx = global_to_local[e]
                    neighbors.append(glob_facet_idx + self.n_cells)

            combined_neigh.append(torch.tensor(neighbors))
            neigh_cents.append(torch.cat(centers))  # [3, 2]
            combined_bc.append(torch.tensor(is_bc)) # [3]
        combined_neigh = torch.stack(combined_neigh).int()
        neigh_cents = torch.stack(neigh_cents)  # [n_cells, 3, 2]
        combined_bc = torch.stack(combined_bc).bool()  # [n_cells, 3]

        # --- Compute gradient vectors in batch ---
        # For each cell, compute the displacement vectors d_i = (neighbor center - cell center)
        center_expanded = centroids.unsqueeze(1)  # shape: [n_cells, 1, 2]
        d = neigh_cents - center_expanded  # shape: [n_cells, 3, 2]
        # Compute weights per neighbor: w_i = 1 / norm(d_i) ** k
        w = 1 / torch.norm(d.double(), dim=2) ** 0.25 # shape: [n_cells, 3]
        # Upweight boundary values
        w[combined_bc] *= 7.5
        w2 = w ** 2  # shape: [n_cells, 3]
        # Compute A = dᵀ @ diag(w²) @ d for each cell.
        # dᵀ has shape [n_cells, 2, 3] and d * w2.unsqueeze(-1) scales each 2D neighbor vector.
        dT = d.transpose(1, 2).double()  # shape: [n_cells, 2, 3]
        A = torch.bmm(dT, d * w2.unsqueeze(-1))  # shape: [n_cells, 2, 2]
        # Invert A for each cell.
        A_inv = torch.inverse(A.double())  # shape: [n_cells, 2, 2]
        # Finally, compute the gradient matrix as A_inv @ dᵀ @ diag(w²)
        # Multiply dᵀ by w2 along the neighbor dimension:
        A_inv_di_T = torch.bmm(A_inv, dT * w2.unsqueeze(1)).float()  # shape: [n_cells, 2, 3]

        # Build gradient matrix
        G_mats = []
        for i in range(2):
            G_mat = build_sparse_gradient_matrix(combined_neigh, A_inv_di_T, i, self.n_cells, self.n_bc_facet)
            G_mats.append(G_mat)

        # Get displacement between cells with facet indexing. In the direction of right to left
        cell_disps, facet_dist_bc = [], []
        for e, cells in facet_to_cell_ord.items():
            if cells.shape[0] == 1:
                # BC cell / facet: Distance from centroid to facet.
                n_hat = normals[e] / torch.norm(normals[e], dim=-1, keepdim=True)
                f = midpoints[e]
                p = centroids[cells[0]]
                disp = n_hat * torch.dot(f - p, n_hat)
                sign = torch.sign(torch.dot(f - p, n_hat))
                dist = torch.norm(disp) * sign
                facet_dist_bc.append(dist)
            else:
                # Main cell / facet: Distance between centroids
                d = centroids[cells[1]] - centroids[cells[0]]
                cell_disps.append(d)
        cell_disps = torch.stack(cell_disps)
        facet_dist_bc = torch.stack(facet_dist_bc)

        return cell_disps, facet_dist_bc, G_mats, combined_neigh


class FVMMesh3D(FVMMesh):
    n_cells: int
    n_facets: int
    n_bc_facet: int

    # Used for FVM calculations
    bc_facet_mask: torch.Tensor  # shape = (n_facet)
    volumes: torch.Tensor  # shape = (n_cells)
    normals: torch.Tensor  # shape = (n_facet, 3)
    facet_areas: torch.Tensor  # shape = (n_facet)
    centroids: torch.Tensor  # shape = (n_cells, 3)
    midpoints: torch.Tensor  # shape = (n_facet, 3)
    facet_vertex: torch.Tensor  # shape = (n_facet, 3, 3)

    def __init__(self, vertices, cells, facets, bc_facet_mask, device="cuda"):
        super().__init__()

        self.vertices = vertices
        self.cells = cells
        self.facets = facets
        self.bc_facet_mask = bc_facet_mask
        self.device = device

        self.n_cells = cells.shape[0]
        self.n_facets = facets.shape[0]
        self.n_bc_facet = bc_facet_mask.sum().item()
        assert facets.shape[0] == bc_facet_mask.shape[0], f'Different number of facets from bc facet mask {facets.shape = }, {bc_facet_mask.shape = }'

        c_print(f'Computing mesh properties', color="bright_magenta")
        self._compute_facet_props(vertices, cells, facets)

    def _compute_facet_props(self, vertices, cells, facets):
        # Compute facet normals and areas
        facet_vertex = vertices[facets]  # shape: [n_facet, 3, 3]
        self.facet_vertex = facet_vertex
        normals, facet_areas = self._facet_normal(facet_vertex)
        self.normals = normals  # shape = [n_facet, 3]
        self.facet_areas = facet_areas  # shape = [n_facet]
        midpoints = torch.mean(facet_vertex, dim=1)  # shape = [n_facet, 3]
        self.midpoints = midpoints

        # Cell volume and centroid
        cell_points = vertices[cells]  # shape: [n_cells, 4, 3]
        self.volumes = self._cell_volume(cell_points)
        self.centroids = torch.mean(cell_points, dim=1)  # shape = [n_cells, 3]

        # Compute mapping of facet to cells
        cell_to_facet, _facet_to_cell, cell_facet_idxs = self._get_cell_facets(cells, facets)  # shape = [n_cells, 4]
        self.cell_to_facet = cell_to_facet

        # Sort cell in order of facet signed direction. ORDER: [-, +], so cell on right comes first.
        self.cell_facet_signs, facet_to_cell, cent_to_facet_disp = self._cell_facet_sign(
            self.centroids, midpoints, cell_to_facet, self.normals, _facet_to_cell, cell_facet_idxs
        )

        # Split tensors into interior and boundary facets
        normals_main, facet_to_cell_main = [], []
        facet_to_cell_bc, normals_bc = [], []
        for e_idx, e_bc in enumerate(self.bc_facet_mask):
            if e_bc:
                facet_to_cell_bc.append(facet_to_cell[e_idx])
                normals_bc.append(normals[e_idx])
            else:
                normals_main.append(normals[e_idx])
                facet_to_cell_main.append(facet_to_cell[e_idx])

        self.normals_main = torch.stack(normals_main)
        self.facet_to_cell_main = torch.stack(facet_to_cell_main)
        self.cent_to_facet_disp = cent_to_facet_disp
        self.facet_to_cell_bc = torch.stack(facet_to_cell_bc).squeeze()

        # Compute grad weighting
        self.cell_grad_stuff = self._grad_weighting(cell_to_facet, facet_to_cell, self.centroids, midpoints, normals)

    def _facet_normal(self, facet_vertex):
        """ Compute outward-facing normals for triangular facets.
            facet_vertex.shape = (n_facet, 3, 3)   [facet_idx, vertex_idx, xyz]
            Returns: normals.shape = (n_facet, 3), facet_areas.shape = (n_facet,)
        """
        # Edge vectors of the triangle
        e1 = facet_vertex[:, 1] - facet_vertex[:, 0]  # shape: [n_facet, 3]
        e2 = facet_vertex[:, 2] - facet_vertex[:, 0]  # shape: [n_facet, 3]
        # Cross product e1 x e2
        normals = torch.cross(e1, e2, dim=1)  # shape: [n_facet, 3]
        facet_areas = 0.5 * torch.norm(normals, dim=1)  # shape: [n_facet]
        return normals, facet_areas

    def _cell_volume(self, vertices):
        """ vertices.shape = (n_cells, 4, 3) """
        a, b, c, d = vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]
        # Compute edge vectors from a
        ab = b - a  # shape [n, 3]
        ac = c - a  # shape [n, 3]
        ad = d - a  # shape [n, 3]
        # Scalar triple product: ab . (ac x ad)
        cross = torch.cross(ac, ad, dim=1)  # shape [n, 3]
        det = torch.sum(ab * cross, dim=1)  # shape [n]
        volume = torch.abs(det) / 6.0
        return volume

    def _get_cell_facets(self, cells, facets):
        """
            Compute which facets belong to each cell.
            cells.shape = (n_cells, 4)  -- tetrahedra
            facets.shape = (n_facet, 3) -- triangular faces
        """
        # 1) Normalize each facet (sort nodes in ascending order).
        facets_sorted, _ = facets.sort(dim=1)  # shape [m, 3]

        # 2) Build a lookup: (nodeA, nodeB, nodeC) -> facet_index
        facet_dict = {}
        for idx, e in enumerate(facets_sorted):
            key = (e[0].item(), e[1].item(), e[2].item())
            facet_dict[key] = idx

        # 3) For each tetrahedron, find the 4 faces.
        # Face patterns for a tetrahedron (consistent winding from create_mesh.py):
        face_patterns = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 1),
            (1, 3, 2),
        ]

        cell_to_facet = []
        for cell in cells:
            v0 = cell[0].item()
            v1 = cell[1].item()
            v2 = cell[2].item()
            v3 = cell[3].item()
            vertices = [v0, v1, v2, v3]

            facet_indices = []
            for fp in face_patterns:
                key = tuple(sorted((vertices[fp[0]], vertices[fp[1]], vertices[fp[2]])))
                facet_indices.append(facet_dict[key])
            cell_to_facet.append(facet_indices)

        cell_to_facet = torch.tensor(cell_to_facet)

        # Compute facet to cell
        unique_facets, inverse = torch.unique(cell_to_facet.reshape(-1), sorted=True, return_inverse=True)

        num_cells, facets_per_cell = self.n_cells, 4
        flat_pos = torch.arange(cell_to_facet.numel(), device=cell_to_facet.device)

        cell_ids = flat_pos // facets_per_cell
        local_facet_ids = flat_pos % facets_per_cell

        order = torch.argsort(inverse)

        sorted_inverse = inverse[order]
        sorted_cells = cell_ids[order]
        sorted_local_facets = local_facet_ids[order]

        counts = torch.bincount(sorted_inverse, minlength=len(unique_facets))

        cell_groups = torch.split(sorted_cells, counts.tolist())
        local_facet_groups = torch.split(sorted_local_facets, counts.tolist())

        facet_to_cell = {
            int(f.item()): cells
            for f, cells in zip(unique_facets, cell_groups)
        }

        cell_facet_idxs = {
            int(f.item()): idxs
            for f, idxs in zip(unique_facets, local_facet_groups)
        }

        return cell_to_facet, facet_to_cell, cell_facet_idxs

    def _grad_weighting(self, cell_to_facet, facet_to_cell_ord, centroids, midpoints, normals):
        """ Use least squares formula to compute gradient weighting (3D).
            grad(u) = A^-1 * b
            A = sum_i w_i^2 (d_i d_i^T)
            b = sum_i w_i^2 d_i (u_i - u_c)
        """
        n_dims = 3
        bound_facet_idxs = torch.nonzero(self.bc_facet_mask, as_tuple=False).flatten()
        global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(bound_facet_idxs)}

        combined_neigh, neigh_cents, combined_bc = [], [], []
        for cell_id, facets in enumerate(cell_to_facet):  # Must keep this order. Neighbor id: torch.cat([Us, Us_bc_facet])
            neighbors, centers, is_bc = [], [], []
            for e in facets:
                e = e.item()
                if len(facet_to_cell_ord[e]) == 2:
                    """ Interior facet"""
                    is_bc.append(False)
                    cells = facet_to_cell_ord[e]
                    neigh_cell = cells[cells != cell_id]
                    centers.append(centroids[neigh_cell])  # [1, 3]
                    neighbors.append(neigh_cell.item())
                else:
                    """ Boundary facet """
                    is_bc.append(True)
                    midpoint = midpoints[e].unsqueeze(0)
                    centers.append(midpoint)
                    glob_facet_idx = global_to_local[e]
                    neighbors.append(glob_facet_idx + self.n_cells)

            combined_neigh.append(torch.tensor(neighbors))
            neigh_cents.append(torch.cat(centers))  # [4, 3]
            combined_bc.append(torch.tensor(is_bc))  # [4]
        combined_neigh = torch.stack(combined_neigh).int()
        neigh_cents = torch.stack(neigh_cents)  # [n_cells, 4, 3]
        combined_bc = torch.stack(combined_bc).bool()  # [n_cells, 4]

        # --- Compute gradient vectors in batch ---
        center_expanded = centroids.unsqueeze(1)  # shape: [n_cells, 1, 3]
        d = neigh_cents - center_expanded  # shape: [n_cells, 4, 3]
        # Compute weights per neighbor: w_i = 1 / norm(d_i) ** k
        w = 1 / torch.norm(d.double(), dim=2) ** 0.25  # shape: [n_cells, 4]
        w[combined_bc] *= 7.5
        w2 = w ** 2  # shape: [n_cells, 4]
        # Compute A = dᵀ @ diag(w²) @ d for each cell.
        dT = d.transpose(1, 2).double()  # shape: [n_cells, 3, 4]
        A = torch.bmm(dT, d * w2.unsqueeze(-1))  # shape: [n_cells, 3, 3]
        # Invert A for each cell.
        A_inv = torch.inverse(A.double())  # shape: [n_cells, 3, 3]
        # Compute gradient matrix as A_inv @ dᵀ @ diag(w²)
        A_inv_di_T = torch.bmm(A_inv, dT * w2.unsqueeze(1)).float()  # shape: [n_cells, 3, 4]

        G_mats = []
        for i in range(n_dims):
            G_mat = build_sparse_gradient_matrix(combined_neigh, A_inv_di_T, i, self.n_cells, self.n_bc_facet)
            G_mats.append(G_mat)

        # Get displacement between cells with facet indexing. In the direction of right to left
        cell_disps, facet_dist_bc = [], []
        for e, cells in facet_to_cell_ord.items():
            if cells.shape[0] == 1:
                # BC cell / facet: Distance from centroid to facet along normal.
                n_hat = normals[e] / torch.norm(normals[e], dim=-1, keepdim=True)
                f = midpoints[e]
                p = centroids[cells[0]]
                disp = n_hat * torch.dot(f - p, n_hat)
                sign = torch.sign(torch.dot(f - p, n_hat))
                dist = torch.norm(disp) * sign
                facet_dist_bc.append(dist)
            else:
                # Main cell / facet: Distance between centroids
                d = centroids[cells[1]] - centroids[cells[0]]
                cell_disps.append(d)
        cell_disps = torch.stack(cell_disps)  # shape: [n_facet_main, 3]
        facet_dist_bc = torch.stack(facet_dist_bc)  # shape: [n_facet_bc]

        return cell_disps, facet_dist_bc, G_mats, combined_neigh


    # _cell_facet_sign is inherited from the base FVMMesh class and is dimension-agnostic.
    # It works unchanged for 3D since it only depends on normals and centroids having matching last dims.
