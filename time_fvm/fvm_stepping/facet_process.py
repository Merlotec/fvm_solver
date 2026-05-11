from typing import TYPE_CHECKING
from cprint import c_print
import torch

from time_fvm.mesh_utils.mesh_store import Facet
from time_fvm.utils.sparse import combine_facet_operators, lift_sparse_matrix, interleave_sparse_rows, to_sparse, SPM
from time_fvm.fvm_stepping.facet_boundary import BoundarySetter
from time_fvm.fvm_stepping.limiter import SlopeLimiter
if TYPE_CHECKING:
    from time_fvm.fvm_equation import PhysicalSetup
    from time_fvm.mesh_utils.fvm_mesh import FVMMesh2D, FVMMesh
    from time_fvm.config_fvm import ConfigFVM


class MeshCache:
    """ Stores unchanging device-tensor mesh geometry, separated from FacetFlux.

        Access CPU-side attributes (centroids, cells, vertices, facets,
        midpoints, volumes, etc.) via mesh.fvm_mesh.xxx for consumers like saving.py.
    """
    device: str
    fvm_mesh: 'FVMMesh'

    # Dimensions
    dim: int
    n_neigh: int        # Numer of facets around each cell
    n_facets: int
    n_cells: int
    n_facets_bc: int

    # General tensors
    normals: torch.Tensor           # (n_facets, dim)
    facet_size: torch.Tensor          # (n_facets, 1)
    normals_hat: torch.Tensor       # (n_facets, dim, 1)
    X_orthog: torch.Tensor          # (n_facets, dim, 1)
    cell_disps: torch.Tensor        # (n_facets, dim)
    facet_to_cell_main: torch.Tensor  # (n_facets_m, 2)
    cell_dist_proj: torch.Tensor     # (n_facets_m)
    cell_facet_signs: torch.Tensor    # (3 * n_cells)  flattened
    cell_to_facet: torch.Tensor       # (3 * n_cells)  flattened
    cent_to_facet_disp: torch.Tensor # (n_cells, 3, dim, 1)

    # BC attributes
    bc_facet_mask: torch.Tensor     # (n_facets)
    bc_locations: torch.Tensor      # (n_facets_bc)
    bc_facet_side: torch.Tensor     # (n_facets_bc)
    facet_to_cell_bc: torch.Tensor   # (n_facets_bc)
    facet_dists_bc: torch.Tensor    # (n_facets_bc, n_comp)

    # Calculations
    G_mats: SPM            # sparse SPM (dim*n_cells, n_cells + n_facets_bc)
    neigh_combine: torch.Tensor     # (n_cells, 2)
    A_face_grad: SPM       # sparse SPM (n_facets*n_comp, n_cells*n_comp)
    b_face_grad: torch.Tensor       # (n_facets*n_comp,)
    flux_mat: SPM                   # (n_cells, n_facets)

    def __init__(self, mesh_setup: FVMMesh, n_comp: int, device: str):
        self.fvm_mesh = mesh_setup
        self.device = device

        self.dim = mesh_setup.dim
        self.n_neigh = self.dim + 1
        self.n_facets = mesh_setup.n_facets
        self.n_cells = mesh_setup.n_cells
        self.n_facets_bc = int(mesh_setup.bc_facet_mask.sum().item())

        # --- Copy tensors to device ---
        self.facet_to_cell_main = mesh_setup.facet_to_cell_main.to(device)
        self.cent_to_facet_disp = mesh_setup.cent_to_facet_disp.to(device).unsqueeze(-1)
        self.cell_facet_signs = (-mesh_setup.cell_facet_signs + 1 / 2).to(torch.int32).view(self.n_neigh * self.n_cells).to(device)    # From {-1, 1} to {0, 1}
        self.cell_to_facet = mesh_setup.cell_to_facet.view(self.n_neigh * self.n_cells).to(device)
        self.facet_to_cell_bc = mesh_setup.facet_to_cell_bc.to(device)
        self.bc_facet_mask = mesh_setup.bc_facet_mask.to(device)

        (cell_disps, facet_dists_bc, G_mats, neigh_combine) = mesh_setup.cell_grad_stuff
        self.facet_dists_bc = facet_dists_bc.to(device).unsqueeze(-1).expand(-1, n_comp)
        G_mats = interleave_sparse_rows(G_mats) # Want output to be easily reshaped.
        self.G_mats = to_sparse(G_mats, device)
        self.neigh_combine = neigh_combine.to(device)

        self.cell_disps = cell_disps.to(device)
        self.normals = mesh_setup.normals.to(device)
        self.facet_size = torch.norm(self.normals, dim=1, keepdim=True).to(device)
        normal_hat = self.normals / self.facet_size
        self.normals_hat = normal_hat.unsqueeze(-1)

        # --- Non-orthogonal correction ---
        cell_disps_full = torch.full((self.n_facets, self.dim), float("nan"), device=device)
        cell_disps_full[~self.bc_facet_mask] = self.cell_disps
        d_cos_theta = (normal_hat * cell_disps_full).sum(dim=1)
        self.cell_dist_proj = d_cos_theta
        X_orthog = cell_disps_full / d_cos_theta.unsqueeze(-1) - normal_hat
        X_orthog[self.bc_facet_mask] = 0                # Boundary edges with neumann BCS are automatically exact.
        self.X_orthog = X_orthog.unsqueeze(dim=-1)

        # --- Boundary facet side tracking ---
        face_assigned = torch.zeros((self.n_facets, self.dim), dtype=torch.bool, device=device)
        face_assigned[self.cell_to_facet, self.cell_facet_signs] = True
        assigned_boundary = face_assigned[self.bc_facet_mask]
        self.bc_facet_side = (~assigned_boundary).float().argmax(dim=1).int()
        self.bc_locations = torch.where(self.bc_facet_mask)[0].int()

        # Flux matrix
        self.flux_mat = self._build_flux_mat()

    def clear_temp(self):
        """ Release device tensors that are no longer needed after initialisation. """
        del self.facet_dists_bc, self.cell_dist_proj, self.facet_to_cell_main, self.cell_disps
        del self.bc_facet_mask, self.flux_mat
        torch.cuda.empty_cache()
        c_print(f'Deleted Mesh temp variables', color="magenta")

    def build_spm_face_grads(self, n_comp: int, dirich_mask: torch.Tensor, neumann_mask: torch.Tensor, dirich_val: torch.Tensor, neumann_val: torch.Tensor):
        """Build sparse operators to compute normal face gradients from cell values.

        Constructs the sparse matrix/operator and offset vector used to compute
        dU/dn on each face from cell-centered values and boundary conditions.
        """
        device = self.device

        n_facets = self.facet_to_cell_main.shape[0]  # number of main facets
        n_cells = self.n_cells
        n_bc = self.n_facets_bc  # number of boundary facets

        """ Main faces"""
        # For each face, we have two contributions.
        # Create row indices: each face i gives two rows (one per contribution).
        rows = torch.arange(n_facets, device=device).repeat_interleave(2)

        # Flatten the cell indices from self.facet_to_cell_main.
        cols = self.facet_to_cell_main.reshape(-1)

        # We want, for each face i, to assign:
        #   - For the first cell (cols entry from self.facet_to_cell_main[i, 0]): -1/d_i
        #   - For the second cell (cols entry from self.facet_to_cell_main[i, 1]): +1/d_i
        #
        # To do this, we first repeat the cell distances for each face:
        cell_dist_rep = self.cell_dist_proj[~self.bc_facet_mask].repeat_interleave(2)  # shape (2*n_facets,)
        # Create a vector with the appropriate signs: first -1 then +1 for each face.
        face_signs = torch.tensor([-1, 1], device=device, dtype=torch.float32).repeat(n_facets)
        # Now compute the nonzero values.
        vals = face_signs / cell_dist_rep

        # Build the sparse matrix A_face.
        A_face = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            size=(n_facets, n_cells))

        A_face_grad_main = lift_sparse_matrix(A_face, n_comp)

        """ Boundary faces """
        # --- Prepare flattened indices for boundary rows ---
        # Each boundary facet gives n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        # Also record the component index for each entry.
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Flatten the condition masks.
        dirich_mask_flat = dirich_mask.reshape(-1)  # True where gradient BC is given as Dirichlet
        neum_mask_flat = neumann_mask.reshape(-1)  # True where gradient BC is Neumann
        # Identify the flattened rows corresponding to Dirichlet gradient entries, with facet index .
        dirich_rows = torch.nonzero(dirich_mask_flat, as_tuple=False).squeeze(1)
        dirich_facet_idx = bc_rows[dirich_rows]
        # Neumann
        neum_rows_all = torch.nonzero(neum_mask_flat, as_tuple=False).squeeze(1)
        neum_comp = comp_idx[neum_rows_all]

        # --- Build the sparse matrix A_grad_bc ---
        # For Dirichlet entries, we want:
        #   coefficient = -1 / facet_dists_bc[facet]  at the column corresponding to
        #   cell = self.facet_to_cell_bc[facet] and component c.
        # For boundary facet i and component c, the cell value is at: col = self.facet_to_cell_bc[i] * n_comp + c
        cols = self.facet_to_cell_bc[dirich_facet_idx] * n_comp + comp_idx[dirich_rows]
        # The coefficient for each Dirichlet entry is -1/facet_dists_bc (for the corresponding boundary facet).
        vals = -1.0 / self.facet_dists_bc[dirich_facet_idx, 0]  # shape: (n_dirich_entries,)
        # The size of the lifted matrix is (n_bc*n_comp, n_cells*n_comp)
        size_grad = (n_bc * n_comp, n_cells * n_comp)
        indices = torch.stack([dirich_rows, cols], dim=0)
        A_grad_bc = torch.sparse_coo_tensor(indices, vals, size=size_grad)

        # --- Build the offset vector b_grad_bc ---
        # For Dirichlet entries:
        #   b = (dirich_val)/facet_dists_bc (applied componentwise)
        # For Neumann entries:
        #   b = neumann_val (applied componentwise)
        b_grad = torch.zeros(n_bc * n_comp, dtype=torch.float32, device=device)
        # Handle Dirichlet gradient entries:
        b_grad[dirich_rows] = dirich_val / self.facet_dists_bc[dirich_facet_idx, 0]
        # Handle Neumann entries:
        b_grad[neum_rows_all] = neumann_val[neum_comp]

        A_face_grad, self.b_face_grad = combine_facet_operators(
            A_face_grad_main, A_grad_bc, b_grad, self.bc_facet_mask, self.n_facets, self.n_cells, n_comp, device
        )

        self.A_face_grad = to_sparse(A_face_grad, device=device)

    def _build_flux_mat(self, dtype=torch.float32):
        """
        Build the incidence matrix T of shape (n_tri, n_facets).
        For each cell i and local facet j, we set:
            T[i, cell_to_facet[i, j]] = cell_facet_sign[i, j].
        """
        c_print("Constricting flux matrix", color="bright_magenta")
        n_cell = self.n_cells  # typically, n_local == 3

        # Create row indices: each cell i contributes n_neigh entries.
        row_indices = torch.arange(n_cell).unsqueeze(1).expand(n_cell, self.n_neigh).reshape(-1)

        # Flatten the edge indices from cell_to_facet for column indices.
        col_indices = self.cell_to_facet.reshape(-1).cpu()
        # Flatten the sign values from tri_edge_sign.
        values = self.cell_facet_signs.reshape(-1).cpu().to(dtype) * 2 - 1
        # Compute the inverse volumes (V_inv is diagonal) and scale the nonzero values.
        volumes_inv = (1.0 / self.fvm_mesh.volumes.cpu()).to(dtype)
        D_values = values * volumes_inv[row_indices]
        # Stack row and column indices for the sparse tensor.
        D_indices = torch.stack([row_indices, col_indices])

        D_shape = [n_cell, self.n_facets]
        flux_mat = torch.sparse_coo_tensor(D_indices, D_values, size=D_shape, device="cpu", dtype=dtype).coalesce()
        # flux_mat = to_csr(flux_mat, self.device)
        flux_mat = to_sparse(flux_mat, device=self.device)
        return flux_mat


class FacetFlux:
    device: str
    dim: int
    n_local: int
    n_facets: int
    n_cells: int
    n_comp: int
    n_facets_bc: int  # cached directly for frequent access
    slope_limiter: SlopeLimiter

    mesh: MeshCache                      # All unchanging device-tensor geometry

    # Boundary condition (BC-specific, not mesh geometry)
    dirich_mask: torch.Tensor       # (n_facets_bc, n_comp)
    neumann_mask: torch.Tensor      # (n_facets_bc, n_comp)
    dirich_val: torch.Tensor
    neumann_val: torch.Tensor
    boundary_setter: BoundarySetter
    bc_type_str: list[str]          # For saving mesh.

    # Temporary / computed variables (set by precompute_shared)
    grad_V: torch.Tensor
    Vs_facet: torch.Tensor
    rho_facet: torch.Tensor
    T_facet: torch.Tensor
    grad_T_n: torch.Tensor
    mom_facet: torch.Tensor         # shape = (n_facets, 2, 2)
    Q_facet: torch.Tensor           # shape = (n_facets, 2, 1)
    phi: torch.Tensor
    cell_grads: torch.Tensor = None
    U_face_all: torch.Tensor = None     # Pre allocated cache

    def __init__(self, phy_setup: 'PhysicalSetup', cfg: 'ConfigFVM', mesh_setup: 'FVMMesh',
                 n_comp: int, bc_tags: dict[int, 'Facet'], device: str = "cpu"):
        self.device = device
        self.cfg = cfg
        self.phy_setup = phy_setup
        self.n_comp = n_comp

        # --- Build device-side mesh ---
        mesh = MeshCache(mesh_setup, n_comp, device)
        self.mesh = mesh
        self.dim, self.n_local = mesh.dim, mesh.n_neigh
        self.n_facets, self.n_facets_bc = mesh.n_facets, mesh.n_facets_bc
        self.n_cells = mesh.n_cells

        self.slope_limiter = SlopeLimiter(mesh_setup.volumes.to(device), cfg)

        # --- Boundary conditions ---
        self._init_bc(bc_tags)
        c_print('_init_bc done', color="magenta")

        # --- Build sparse face-gradient operators ---
        mesh.build_spm_face_grads(self.n_comp, self.dirich_mask, self.neumann_mask, self.dirich_val, self.neumann_val)

        # Decide which components gradients are needed:
        # Vx, Vy, (Vz), rho, T
        self.grad_comps = list(range(self.dim)) + [self.dim+1]

        c_print('Complete init FVMEdgeInfo', color="magenta")

    def clear_temp(self):
        self.mesh.clear_temp()
        del self.dirich_val, self.neumann_val
        del self.dirich_mask, self.neumann_mask
        torch.cuda.empty_cache()
        c_print('Deleted temp variables', color="magenta")

    def _init_bc(self, bc_tags: dict[int, Facet]):
        mesh = self.mesh

        bc_type_str = []
        dirich_mask, neumann_mask = [], []
        dirich_val, neumann_val = [], []
        farfield_mask, inlet_mask = [], []
        for bc_idx, e_type in bc_tags.items():
            dirich_mask.append(e_type.dirichlet())
            neumann_mask.append(e_type.neumann())
            dirich_val.append(e_type.U)
            neumann_val.append(e_type.dUdn)

            farfield_mask.append(e_type.farfield())
            inlet_mask.append(e_type.inlet())

            bc_type_str.append(e_type.tag)

        self.bc_type_str = bc_type_str
        self.dirich_mask, self.neumann_mask = torch.tensor(dirich_mask, device=self.device), torch.tensor(neumann_mask, device=self.device)
        dirich_val, neumann_val = torch.tensor(dirich_val, dtype=torch.float32, device=self.device), torch.tensor(neumann_val, dtype=torch.float32, device=self.device)
        self.dirich_val = dirich_val[self.dirich_mask]
        self.neumann_val = neumann_val[self.neumann_mask]

        # Farfield and inlet boundary conditions
        self.farfield_mask = torch.tensor(farfield_mask, device=self.device)     # shape = (n_facets_bc)
        self.inlet_mask = torch.tensor(inlet_mask, device=self.device)     # shape = (n_facets_bc)

        assert self.dirich_mask.shape[0] == mesh.bc_facet_mask.sum(), f'Wrong mask shape'
        assert self.neumann_mask.shape[0] == mesh.bc_facet_mask.sum(), f'Wrong mask shape'
        assert self.farfield_mask.shape[0] == mesh.bc_facet_mask.sum(), f'Wrong mask shape'
        assert self.inlet_mask.shape[0] == mesh.bc_facet_mask.sum(), f'Wrong mask shape'

        self.boundary_setter = BoundarySetter(self, self.phy_setup)

    def compute_facet(self, Us):
        """ Precompute shared values that are used multiple times later.
            Us.shape = [n_cells, n_component] """
        mesh = self.mesh

        U_facet_bc = self.boundary_setter.set_face_values(Us, self.cell_grads)      # shape = [n_facets_bc, n_comp]
        Us_cell_facet = torch.cat([Us, U_facet_bc])         # shape = [n_cells + n_facets_bc, n_comp]
        cell_grads = self._cell_grads(Us_cell_facet)        # shape = [n_cells, 2, n_comp]

        # Compute limited face values
        Us_face, phi_lim = self._limit_face_vals(Us, Us_cell_facet, cell_grads)   # Us_face.shape = [n_cells, 3, n_comp], phi_lim.shape =  [n_cells, 1, n_comp]
        Us_face = Us_face.view(self.n_local * self.n_cells, self.n_comp)
        cell_grads = cell_grads * phi_lim
        self.cell_grads = cell_grads    # Save for boundary conditions

        # Project to left and right face values
        U_face_all = torch.empty((self.n_facets, 2, self.n_comp), device=self.device)  # shape = [n_facets, 2, n_comp+6]
        U_face_all[mesh.cell_to_facet, mesh.cell_facet_signs] = Us_face
        U_face_all[mesh.bc_locations, mesh.bc_facet_side] = U_facet_bc

        # Decompose components back
        self.Vs_facet = U_face_all[:, :, :self.dim].contiguous()                   # shape = [n_facets, facets=2, n_comp=2]
        self.rho_facet = U_face_all[:, :, self.dim].unsqueeze(-1).contiguous()     # shape = [n_facets, facets=2, dims=1]
        self.T_facet = U_face_all[:, :, self.dim+1].unsqueeze(-1).contiguous()     # shape = [n_facets, facets=2, dims=1]
        # Conserved quantities
        self.mom_facet, _, self.Q_facet = self.phy_setup.primatives_to_state(self.Vs_facet, self.rho_facet, self.T_facet)
        self.phi = (self.Vs_facet * mesh.normals.unsqueeze(1)).sum(dim=-1) # shape = [n_facets, facets=2, ]

        # Compute face and cell gradients of velocity and temperature
        grad_F = cell_grads[:, :, self.grad_comps].reshape(self.n_cells, 6)      # shape = [n_cells, {dx, dy} * {vx, vy, T}]
        grad_F_flat = torch.repeat_interleave(grad_F, 3, dim=0, output_size=3*self.n_cells) # [dvx/dx, dvy/dx, dvx/dy, dvy/dy, dT/dx, dT/dy]
        grad_F_bc = grad_F[mesh.facet_to_cell_bc]
        # Project cell gradients to left and right face values
        grad_facet_all = torch.empty((self.n_facets, 2, 6), device=self.device)
        grad_facet_all[mesh.cell_to_facet, mesh.cell_facet_signs] = grad_F_flat
        grad_facet_all[mesh.bc_locations, mesh.bc_facet_side] = grad_F_bc
        # Average gradient over both sides of facet
        grad_F_lstsq = grad_facet_all.mean(dim=1)
        grad_F_lstsq = grad_F_lstsq.reshape(self.n_facets, self.dim, self.dim+1)   # shape = [n_facets, {x, y}, {vx, vy, T}]

        # Get face gradients using direct cell to face gradient operator.
        grad_faces_n = self._face_grads(Us)                             # shape = [n_faces, n_comp]
        grad_F_dn = grad_faces_n[:, self.grad_comps]                    # shape = [n_faces, 3]

        # Non-orthogonal correction for facet gradient: n . grad(U)_f = C du + (n - C d) . grad(U)_f
        # Remove parallel part of gradient
        dFdn_correct = grad_F_dn - (grad_F_lstsq * mesh.X_orthog).sum(dim=1)      # shape = [n_facets, 3]
        # Replace normal part of gradient with facet gradient
        grad_F_dot_n = (grad_F_lstsq * mesh.normals_hat).sum(dim=1, keepdim=True)       # shape = [n_facets, 1, 3]
        grad_F = grad_F_lstsq + (dFdn_correct.unsqueeze(1) - grad_F_dot_n) * mesh.normals_hat            # [n_facets, 2, 3]

        self.grad_V = grad_F[:, :, :self.dim]
        self.grad_T_n = dFdn_correct[:, -1]      # shape = [n_facets]

    def _limit_face_vals(self, Us, Us_cell_facet, cell_grads):
        """ Limited B-J scheme for cell to face interpolation.
        """
        mesh = self.mesh
        U_cent = Us.unsqueeze(1)        # shape = [n_cells, 1, n_comp]
        Us_neigh = Us_cell_facet[mesh.neigh_combine]  # shape = [n_cells, neigh=3, n_comp]

        # Uncorrected update
        grads = cell_grads.unsqueeze(1)     # shape = [n_cells, 1, dims=2, n_comp]
        dU = (grads * mesh.cent_to_facet_disp).sum(dim=2)  # shape = [n_cells, neigh=3, n_comp]

        # Select limiting neighbor values and compute gradient limiter
        U_cent_neigh = torch.cat([U_cent, Us_neigh], dim=1)             # shape = [n_cells, 4, n_comp]
        U_upper = torch.amax(U_cent_neigh, dim=1, keepdim=True) - U_cent      # shape = [n_cells, 1, n_comp]
        U_lower = torch.amin(U_cent_neigh, dim=1, keepdim=True) - U_cent

        numerator = torch.where(dU > 0, U_upper, U_lower)           # shape = [n_cells, neigh=3, n_comp]
        phi_lim = self.slope_limiter.limit(numerator, dU)           # shape = [n_cells, neigh=3, n_comp]
        Us_face = torch.addcmul(U_cent, dU, phi_lim)                # shape = [n_cells, neigh=3, n_comp]

        return Us_face, phi_lim

    def _cell_grads(self, Us_cell_face):
        """ Vectorised gradient computation
            Gradient = G @ (u_neigh - u_cell)
            Us_cell_face.shape = (n_cells+n_facets_bc, N_component)
            Returns: Gradient matrix of shape (n_cells, 2, N_component)
        """
        # combined_grad = torch.mm(self.mesh.G_mats, Us_cell_face)  # combined_grad.shape == [n_cells*dim, N_component]
        combined_grad = self.mesh.G_mats.spMM(Us_cell_face)  # combined_grad.shape == [n_cells*dim, N_component]
        cell_grads = combined_grad.view(self.n_cells, self.dim, self.n_comp)    # shape = [n_cells, dim, N_component]
        return cell_grads

    def _face_grads(self, Us):
        """ n . grad(U) on faces.

            Us.shape = (n_cells, N_component)
            Returns: shape = [n_facets, N_component]
        """
         # U_centroid = Us[self.edge_to_tri_main]      # shape = [n_facets, 2, N_component]
        # dU = U_centroid[:, 1] - U_centroid[:, 0]        # shape = [n_facets, N_component]
        # dUdn_face = torch.empty((self.n_facets, self.n_comp), device=self.device)
        # dUdn_face_m = dU / self.cell_dist.unsqueeze(-1)       # shape = [n_facets, N_component]
        # # print(f'{dUdn_face.shape = }, {self.bc_edge_mask.shape = }')
        # dUdn_face[~self.bc_edge_mask] = dUdn_face_m
        # #
        # # On boundary. Either u or du/dn is given
        # # Neumann: n.grad(u) = du/dn
        # dUdn_face[self.neum_all] = self.neumann_val
        #
        # # Dirichlet: n.grad(u) = 1/d * (u_bc - u)
        # u_centroid_bc = Us[self.edge_to_tri_bc]  # shape = [n_bc_edges, N_component]
        # U_cent_bc_dir = u_centroid_bc[self.dirich_mask]     # shape = [n_dirich_edges]
        # edge_dists = self.edge_dists_bc[self.dirich_mask]    # shape = [n_dirich_edges]
        # dudn_face_bc_dir = (self.dirich_val - U_cent_bc_dir) / edge_dists
        # dUdn_face[self.dirich_all] = dudn_face_bc_dir

        # """ SPARSE"""
        mesh = self.mesh
        # dUdn_face_flat = torch.addmv(mesh.b_face_grad, mesh.A_face_grad, Us.flatten())
        dUdn_face_flat = mesh.A_face_grad.spMV(Us.flatten(), mesh.b_face_grad)
        dUdn_face = dUdn_face_flat.view(self.n_facets, self.n_comp)

        return dUdn_face
