from typing import TYPE_CHECKING
import torch

from time_fvm.config_fvm import ConfigFVM, ConfigBC, BCMode
from time_fvm.utils.sparse import to_sparse, SPM

if TYPE_CHECKING:
    from torch import Tensor
    from time_fvm.fvm_stepping.facet_process import FacetCalc
    from time_fvm.fvm_equation import FluidConstitution


class BoundarySetter:
    """ Non-orthogonal correction for Neumann BCs."""
    dim: int
    n_comp: int
    n_facets_bc: int
    n_cells: int

    grad_comps: torch.Tensor          # shape = [n_neum_edges, 2, 1]
    where_neum: tuple[torch.Tensor]      # shape = [2][n_neum_edges, 2]

    # Matrices for general face values
    A_bc: SPM
    b_bc: torch.Tensor
    A_corr: SPM                 # Non orthogonal correction matrix

    # Merged BC groups, one per unique BCMode
    bc_groups: dict  # {BCMode: (BC, cell_indices: Tensor)}

    def __init__(self, E_props: FacetCalc, phy_setup: FluidConstitution):
        self.device = E_props.device
        self.phy_setup = phy_setup

        self.bc_groups: dict = {}  # finalized: {BCMode: (BC, cell_indices)}
        _bc_specs: dict = {}  # local dict to store specs before merging

        self.dim, self.n_local = E_props.dim, E_props.n_local
        self.n_comp = E_props.n_comp
        self.n_facets_bc = E_props.n_facets_bc
        self.n_cells = E_props.n_cells

        mesh = E_props.mesh

        # Get boundary condition setup
        if torch.any(E_props.farfield_mask).item():
            exit_cell2facet = mesh.facet_to_cell_bc[E_props.farfield_mask]
            ff_facet_sign = 2 * (mesh.bc_facet_side[E_props.farfield_mask] - 0.5)
            ff_facet_normals = mesh.normals_hat.squeeze()[mesh.bc_facet_mask][E_props.farfield_mask]
            ff_facet_normals = ff_facet_normals * ff_facet_sign.unsqueeze(-1)
            _bc_specs.setdefault(E_props.cfg.exit_cfg.mode, []).append(
                (E_props.cfg, E_props.cfg.exit_cfg, E_props.farfield_mask, exit_cell2facet, ff_facet_normals)
            )
        if torch.any(E_props.inlet_mask).item():
            inlet_cell2facet = mesh.facet_to_cell_bc[E_props.inlet_mask]
            inlet_facet_sign = 2 * (mesh.bc_facet_side[E_props.inlet_mask] - 0.5)
            inlet_facet_normals = mesh.normals_hat.squeeze()[mesh.bc_facet_mask][E_props.inlet_mask]
            inlet_facet_normals = inlet_facet_normals * inlet_facet_sign.unsqueeze(-1)
            _bc_specs.setdefault(E_props.cfg.inlet_cfg.mode, []).append(
                (E_props.cfg, E_props.cfg.inlet_cfg, E_props.inlet_mask, inlet_cell2facet, inlet_facet_normals)
            )

        # _finalize_bc_groups logic
        for mode, specs in _bc_specs.items():
            cfg = specs[0][0]  # ConfigFVM, same for all specs in a group
            bc_specs = [(bc_cfg, bc_mask, bc_normals) for _, bc_cfg, bc_mask, _, bc_normals in specs]
            bc = BC(self.phy_setup, cfg, mode, bc_specs, self.n_facets_bc)

            # Merge cell_indices: concat + reorder to align with bc.bc_idx
            face_to_pos = torch.full((self.n_facets_bc,), -1, dtype=torch.long, device=self.device)
            offset = 0
            for _, _, bc_mask, _, _ in specs:
                idx = torch.where(bc_mask)[0]
                face_to_pos[idx] = torch.arange(offset, offset + idx.shape[0], device=self.device)
                offset += idx.shape[0]
            perm = face_to_pos[bc.bc_idx]
            merged_cells = torch.cat([ci for _, _, _, ci, _ in specs])[perm]

            self.bc_groups[mode] = (bc, merged_cells)

        cell_to_facet = mesh.cell_to_facet.view(-1, self.n_local)

        # Flatten out all Neumann BCs and index according to order where_neum_all[0]
        neum_mask_all = torch.zeros_like(mesh.bc_facet_mask)
        neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, self.n_comp)
        neum_mask_all[mesh.bc_facet_mask] = E_props.neumann_mask
        where_neum_all = torch.where(neum_mask_all)
        where_neum = {'facet': where_neum_all[0], 'comp': where_neum_all[1]}  # shape = [n_neum_facets, 2]
        # Mapping from boundary id to boundary facet id
        self.where_neum = torch.where(E_props.neumann_mask)

        # Mapping from boundary facet to cell
        bc_facet_to_cell = torch.zeros_like(mesh.bc_facet_mask).long()
        bc_facet_to_cell[mesh.bc_facet_mask] = mesh.facet_to_cell_bc

        # Cells corresponding to Neumann BC
        self.neum_cells = bc_facet_to_cell[where_neum_all[0]]  # shape = [n_neum_facets], which cells have neuman BCs
        where_neum['cells'] = self.neum_cells
        # Facet within cell corresponding to Neumann BC
        cell_facet_num = (where_neum['facet'].unsqueeze(-1).repeat(1, self.n_local) == cell_to_facet[where_neum['cells']])
        cell_facet_id = torch.where(cell_facet_num)[1]
        where_neum['cell_facet_id'] = cell_facet_id

        # Which component of gradient is needed for Neumann BC
        self.grad_comps = where_neum['comp'].unsqueeze(1).repeat(1, self.dim).unsqueeze(2)     # shape = [n_neum_facets, self.dim, 1]
        # Normal vector of facets
        n_hats = mesh.normals_hat.squeeze()[where_neum['facet']]  # shape = [n_neum_facets, 2]
        # Displacement from centroid to facet
        cent_to_facet = mesh.cent_to_facet_disp[where_neum['cells']].squeeze()  # shape = [n_neum_facet, 3, 2]
        r = cent_to_facet[torch.arange(cent_to_facet.shape[0]), where_neum['cell_facet_id']]  # shape = [n_neum_facet, 2]
        # Normal component of r
        d = n_hats * (r * n_hats).sum(dim=1, keepdim=True)  # shape = [n_neum_facet, 2]
        # Parallel component of r
        self.l = r - d

        # Precompute sparse matrices
        self.A_bc, self.b_bc = self._build_spm_face_vals(E_props)
        self.A_corr = self._build_sparse_nonorthog_correct()

        # Clean up unused stuff
        del self.neum_cells, self.where_neum, self.grad_comps, self.l

    def set_face_values(self, Us, cell_grads=None):
        """Compute and return boundary face values from cell values.

        Uses the precomputed sparse operator to map flattened cell values to
        boundary face values, then applies non-orthogonal correction and
        BC adjustments — one call per unique BCMode.
        """
        # Boundary face values
        U_face_flat = self.A_bc.spMV(Us.flatten(), self.b_bc)      # shape = [n_facets_bc * n_comp]

        # Non-orthogonal correction for boundary values, if gradients exist.
        if cell_grads is not None:
            self._non_orthogonal_correction(U_face_flat, cell_grads)

        # Reshape back to (n_facets_bc, n_comp)
        U_face = U_face_flat.view(self.n_facets_bc, self.n_comp)

        # Apply BCs: one call per unique BCMode (merged masks vectorised together)
        for bc_calc, cell_indices in self.bc_groups.values():
            bc_calc.set_bc_U_face(U_face, Us[cell_indices])

        return U_face

    def _non_orthogonal_correction(self, U_face_flat, cell_grads):
        """
        Use previous gradient for non-orthogonal correction.

        U_face.shape = [n_bc_faces, n_comp]
        cell_grads.shape = [n_cells, 2, n_comp]

        r = centroid to midpoint.
        d = normal component of r
        l = r - d, parallel component of r
        U_f = U_0 + d * dUdn + (r-d) grad(U)
        """
        # dU = torch.mv(self.A_corr, cell_grads.reshape(-1))
        # U_face_flat.add_(dU)

        # U_face_flat.addmv_(
        #     self.A_corr,
        #     cell_grads.view(-1),
        # )

        dU = self.A_corr.spMV(cell_grads.view(-1))
        U_face_flat.add_(dU)

    def _build_sparse_nonorthog_correct(self):
        """
        Builds sparse A and b such that:

            U_face_flat += A @ cell_grads_flat + b

        with:
            U_face_flat.shape     = [n_bc_faces * n_comp]
            cell_grads_flat.shape = [n_cells * 2 * n_comp]
        """
        device = self.device
        n_comp = self.n_comp
        n_facets_bc = self.n_facets_bc
        n_cells = self.n_cells

        # Number of Neumann correction entries
        n_neum = self.neum_cells.shape[0]

        # Output rows: entries in flattened U_face
        # where_neum[0] = face index
        # where_neum[1] = component index
        face_ids = self.where_neum[0].to(device)
        face_comps = self.where_neum[1].to(device)

        row_ids = face_ids * n_comp + face_comps  # [n_neum]

        # Each correction uses spatial_dirs spatial gradient components
        spatial_dirs = torch.arange(self.dim, device=device)  # [0, 1] for 2D, [0, 1, 2] for 3D

        # Expand rows for x/y(/z) gradient contributions
        rows = row_ids[:, None].expand(n_neum, self.dim).reshape(-1)

        # Cell ids used by Neumann faces
        cell_ids = self.neum_cells.to(device)

        # grad_comps should select the component of the gradient being used.
        # Usually this is the same as face_comps, but this follows your original gather logic.
        grad_comps = self.grad_comps.to(device)

        # Make grad_comps shape [n_neum, self.dim]
        if grad_comps.ndim == 3:
            grad_comps = grad_comps.squeeze(-1)
        elif grad_comps.ndim == 1:
            grad_comps = grad_comps[:, None].expand(n_neum, self.dim)

        # Column ids into flattened cell_grads:
        # cell_grads[cell, spatial_dir, comp]
        # flat index = cell * (self.dim * n_comp) + spatial_dir * n_comp + comp
        cols = (
                cell_ids[:, None] * (self.dim * n_comp)
                + spatial_dirs[None, :] * n_comp
                + grad_comps
        ).reshape(-1)

        # Coefficients are self.l
        vals = self.l.to(device=device).reshape(-1)

        A = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols], dim=0),
            values=vals,
            size=(n_facets_bc * n_comp, n_cells * self.dim * n_comp),
            device=device,
        ).coalesce()

        # A = to_csr(A, device=device)
        A = to_sparse(A, device=device)
        return A

    def _build_spm_face_vals(self, E_props: FacetCalc):
        """ Compute bc facet values using sparse matrix multiplication. """

        device = E_props.device
        n_bc = E_props.n_facets_bc
        n_comp = E_props.n_comp
        n_cells = E_props.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary facet gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = E_props.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = E_props.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.facet_to_cell_bc[ facet_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary facet for each flattened row.
        A_cols = E_props.mesh.facet_to_cell_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A)# .coalesce().to_sparse_csr()
        # A_bc = to_csr(A_bc, device=device)
        A_bc = to_sparse(A_bc, device=device)
        # Build the offset vector b.
        b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        b_bc[dirich_mask] = E_props.dirich_val
        # For Neumann entries, add the offset computed from the facet distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        b_bc[neum_mask] = E_props.neumann_val[comp_idx[neum_mask]] * E_props.mesh.facet_dists_bc.flatten()[neum_mask]

        return A_bc, b_bc


class BC:
    dim: int
    bc_normals: Tensor      # shape = [n_bc, dim]
    v_n_inf: Tensor         # shape = [n_bc]
    v_t_inf: Tensor         # shape = [n_bc, dim]

    def __init__(self, phy_setup: FluidConstitution, cfg: ConfigFVM, mode: BCMode,
                 specs: list, n_facets_bc: int):
        """
        Build a merged BC from multiple same-mode boundary regions.

        Args:
            phy_setup: EOS and physics functions.
            cfg: global configuration.
            mode: the BCMode shared by all specs.
            specs: list of (bc_cfg: ConfigBC, bc_mask: Tensor, bc_normals: Tensor).
                   bc_mask has shape [n_facets_bc]; bc_normals has shape [n_region, 2].
            n_facets_bc: total number of boundary facets.
        """
        self.cfg = cfg
        self.device = cfg.device
        self.phy_setup = phy_setup
        self.dim = phy_setup.dim

        # --- Merge masks ---
        merged_mask = specs[0][1].clone()
        for _, bc_mask, _ in specs[1:]:
            merged_mask = merged_mask | bc_mask
        self.bc_idx = torch.where(merged_mask)[0]
        self.n_bc = self.bc_idx.shape[0]

        # --- Build per-face parameter vectors ---
        # Concatenate per-region tensors, then reorder to align with bc_idx.
        face_to_pos = torch.full((n_facets_bc,), -1, dtype=torch.long, device=self.device)
        offset = 0
        for _, bc_mask, _ in specs:
            idx = torch.where(bc_mask)[0]
            face_to_pos[idx] = torch.arange(offset, offset + idx.shape[0], device=self.device)
            offset += idx.shape[0]
        perm = face_to_pos[self.bc_idx]                     # [n_merged]

        def _cat_reorder(lst):
            return torch.cat(lst)[perm]

        normals_list = []
        T_inf_list, rho_inf_list, v_n_inf_list, v_t_inf_list = [], [], [], []
        p_inf_list = []
        R_m_list, R_p_list, S_list = [], [], []
        if mode == BCMode.Farfield:
            self.R = phy_setup.R
            self.gamma = phy_setup.gamma

        for bc_cfg, bc_mask, bc_normals in specs:
            n = int(bc_mask.sum().item())
            normals_list.append(bc_normals)

            T_inf = torch.full((n,), bc_cfg.T_inf, device=self.device)
            rho_inf = torch.full((n,), bc_cfg.rho_inf, device=self.device)
            T_inf_list.append(T_inf)
            rho_inf_list.append(rho_inf)
            if self.dim == 2:
                tangents = torch.stack((-bc_normals[:, 1], bc_normals[:, 0]), dim=1)
                v_t_inf = tangents * bc_cfg.v_t_inf
                v_t_inf_list.append(v_t_inf)
                # v_n_inf is negated: bc_cfg.v_n_inf is inward-positive, we store outward-positive
                v_n_inf_list.append(torch.full((n,), -bc_cfg.v_n_inf, device=self.device))

            else:
                # 3D, get tangential and normal component directly.
                v_inf = torch.tensor(bc_cfg.v_inf, device=self.device)
                v_n_inf = (bc_normals * v_inf).sum(dim=1)
                v_t_inf = v_inf - bc_normals * v_n_inf.unsqueeze(-1)
                v_t_inf_list.append(v_t_inf)
                v_n_inf_list.append(v_n_inf)

            if mode == BCMode.Characteristic:
                a_inf, p_inf = phy_setup.eos_c(rho_inf, T_inf), phy_setup.eos_P(rho_inf, T_inf)
                p_inf_list.append(p_inf)
            elif mode == BCMode.Farfield:
                a_inf, p_inf = phy_setup.eos_c(rho_inf, T_inf), phy_setup.eos_P(rho_inf, T_inf)
                v_n = torch.full((n,), bc_cfg.v_n_inf, device=self.device)
                # R_m_far = (-v_n_inf) - 2a/(γ-1), R_p_far = (-v_n_inf) + 2a/(γ-1)
                R_m_list.append(-v_n - 2 * a_inf / (self.gamma - 1))
                R_p_list.append(-v_n + 2 * a_inf / (self.gamma - 1))
                S_list.append(p_inf / (rho_inf ** self.gamma))

        self.bc_normals = _cat_reorder(normals_list)
        self.T_inf = _cat_reorder(T_inf_list)
        self.rho_inf = _cat_reorder(rho_inf_list)
        self.v_n_inf = _cat_reorder(v_n_inf_list)       # shape = [n_bc]
        self.v_t_inf = _cat_reorder(v_t_inf_list)       # shape = [n_bc, dim]

        match mode:
            case BCMode.Characteristic:
                self.set_bc_U_face = self.BC_characteristic
                self.p_inf = _cat_reorder(p_inf_list)
            case BCMode.Farfield:
                self.set_bc_U_face = self.BC_farfield
                self.R_m_far = _cat_reorder(R_m_list)
                self.R_p_far = _cat_reorder(R_p_list)
                self.S_far = _cat_reorder(S_list)
            case _:
                raise NotImplementedError(f"Unknown BCMode {mode}")

    def set_bc_U_face(self, U_face, Us_bc_cells):
        """ Set U_face.
            U_face: shape = [n_facets, n_comp], all boundary facets. Set value in place, given by mask.
            Us_bc_cells: shape = [n_bc_facets, n_comp], cell values at boundary facets.
        """
        raise NotImplementedError

    def _split_Vs(self, Vs):
        """ Split velocity into normal and tangential components.
            Vs: shape = [n_bc, 2]
            Return V_n, V_t: shape = [n_bc], [n_bc, dim]
        """
        n = self.bc_normals

        v_n = (Vs * n).sum(dim=1)
        v_t = Vs - v_n.unsqueeze(-1) * n

        return v_n, v_t

    def _recombine_Vs(self, v_n, v_t):
        """ Combine normal and tangential component of Vs back into x-y components."
            V_n: shape = [n_bc]
            V_t: shape = [n_bc]
            Return V: shape = [n_bc, dim]
        """
        n = self.bc_normals

        V = v_n.unsqueeze(-1) * n + v_t

        return V

    def _gating(self, v_n_int, c_int):
        """ Gating for forward and backward characteristics.
            v_n_int.shape = [n_bc]
            return.shape = 3, [n_bc]
        """
        lambda_0 = v_n_int - c_int
        lambda_1 = v_n_int
        lambda_2 = v_n_int + c_int

        scale = 10.0 / c_int.mean()

        g0 = 0.5 * (1.0 - torch.tanh(scale * lambda_0))
        g1 = 0.5 * (1.0 - torch.tanh(scale * lambda_1))
        g2 = 0.5 * (1.0 - torch.tanh(scale * lambda_2))

        return g0, g1, g2

    # ------------------------------- Specific BC implementations -------------------------------
    def BC_characteristic(self, U_face, Us_bc_cells):
        """ Characteristic BC.
            Use W = (rho, v_n, p) -> dW/dt + div(f(W)) = 0
            Linearize using W = W_int + delta W
            Diagonalise equations
            Then solve for delta W_b by continuing characteristics from the left and right side.
            Tangential velocity is interpolated as well.

            U_face.shape = [n_bc_facets, n_comp], all boundary facets. Set value in place, given by mask.
            Us_bc_cells.shape = [n_inlet, n_comp], cell values at boundary facets.
        """
        # 1) Interior properties
        Vs = Us_bc_cells[:, :self.dim]
        rho_int = Us_bc_cells[:, self.dim]
        T_int = Us_bc_cells[:, self.dim+1]

        v_n_int, v_t_int = self._split_Vs(Vs)

        p_int = self.phy_setup.eos_P(rho_int, T_int)
        c_int = self.phy_setup.eos_c(rho_int, T_int)

        # 2) Exterior - broadcasted vectors
        drho = self.rho_inf - rho_int
        dvn = self.v_n_inf - v_n_int
        dp = self.p_inf - p_int

        # 3) Project dW into characteristic variables:
        """R =
            [ 1         1   1       ]
            [ -c/rho    0   c/rho   ]
            [ c²        0   c²      ]
            R⁻¹ =
            [ 0     -rho/(2c)   1/(2c²)]
            [ 1     0           1/c²   ]
            [ 0     rho/(2c)    1/(2c²)]
        """
        # dChi = R_inv @ dW
        half = 0.5
        inv_c = 1 / c_int
        inv_c2 = inv_c ** 2
        half_invc2_dp = half * inv_c2 * dp
        half_rho_invc_dvn = half * rho_int * inv_c * dvn

        dchi_0 = -half_rho_invc_dvn + half_invc2_dp
        dchi_1 = drho - inv_c2 * dp
        dchi_2 = half_rho_invc_dvn + half_invc2_dp

        # 4) Gating for forward and backward components, smoothly
        g0, g1, g2 = self._gating(v_n_int, c_int)
        dchi_0 = g0 * dchi_0
        dchi_1 = g1 * dchi_1
        dchi_2 = g2 * dchi_2

        # 4.1) Tangential velocity interpolation
        g1 = g1.unsqueeze(1)
        v_t_b = g1 * v_t_int + (1.0 - g1) * self.v_t_inf

        # 5) Reconstruct dW = R @ dChi
        c2 = c_int * c_int
        c_over_rho = c_int / rho_int

        d_rho = dchi_0 + dchi_1 + dchi_2
        d_vn = -c_over_rho * dchi_0 + c_over_rho * dchi_2
        d_p = c2 * (dchi_0 + dchi_2)
        # W -> W + dW
        rho_b = rho_int + d_rho
        v_n_b = v_n_int + d_vn
        p_b = p_int + d_p

        # 6) Recombine velocity
        Vs = self._recombine_Vs(v_n_b, v_t_b)

        # 7) EOS back to temperature
        T_b = self.phy_setup.eos_T(rho_b, p_b)

        U_face_b = torch.cat((Vs, rho_b.unsqueeze(-1), T_b.unsqueeze(-1)), dim=-1)
        U_face[self.bc_idx] = U_face_b

    def BC_farfield(self, U_face, Us_bc_cells):
        """Set farfield boundary conditions using blended characteristic approach. Good if flow changes direction.

        This function implements a smooth, blended farfield boundary condition that
        automatically transitions between inflow and outflow using Riemann invariants
        and entropy.

        Theory - Riemann Invariants:
        ----------------------------
        1. Left-running Riemann invariant (characteristic speed: u - a):
           R⁻ = u - 2a/(γ-1)
        2. Entropy invariant (characteristic speed: u):
           S = p/ρ^γ = RT/ρ^(γ-1)
           constant along streamlines for isentropic flow
        3. Right-running Riemann invariant (characteristic speed: u + a):
           R⁺ = u + 2a/(γ-1)

        Blending Strategy:
        ------------------
        The method smoothly interpolates each invariant based on the sign and magnitude  of λ = u ± a:
        - If λᵢ >> 0 (strongly outgoing): use interior value (information flows outward)
        - If λᵢ << 0 (strongly incoming): use farfield value (information flows inward)
        - If λᵢ ≈ 0 (near-sonic): blend smoothly between interior and farfield
        Blending function:
            αᵢ = 0.5 * (1 - tanh(λᵢ/c))
        Blended invariants:
            R⁻_bc = α₁ * R⁻_far + (1-α₁) * R⁻_int
            S_bc  = α₂ * S_far  + (1-α₂) * S_int
            R⁺_bc = α₃ * R⁺_far + (1-α₃) * R⁺_int

        Reconstruction:
        ---------------
        Once the blended invariants are computed, the primitive variables are
        reconstructed:

        1. Normal velocity:
           u_bc = (R⁺_bc + R⁻_bc) / 2
        2. Speed of sound:
           a_bc = (γ-1)/4 * (R⁺_bc - R⁻_bc)
        3. Density (from entropy and speed of sound):
           ρ_bc = (a²_bc / (γ * S_bc))^(1/(γ-1))
        4. Temperature (from equation of state):
           T_bc = a²_bc / (γ * R)

        Parameters:
        -----------
        U_face : torch.Tensor, shape [n_facets_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_farfield_facets, n_comp]
            Interior cell values adjacent to farfield facets [V_x, V_y, ρ, T]
        dt : float, optional
            Time step (unused, kept for interface compatibility)

        Notes:
        ------
        - The tangential velocity component is always preserved from interior
        - The blending scale is O(a), making the transition region sonic-scale
        - Farfield values R⁻_far, R⁺_far, S_far are precomputed from exit conditions
        - This approach is stable for both subsonic and supersonic flows
        """
        gm1 = self.gamma - 1

        # Interior properties
        V_int = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_facet, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_facet]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_facet]
        # Parallel and tangential velocity
        V_n, V_t = self._split_Vs(V_int)

        # Interior invariants:
        # a_int = self.phy_setup.eos_c(rho_int, T_int)
        a_int = self.phy_setup.eos_c(rho_int, T_int)
        R_m_int = V_n - 2 * a_int / gm1                     # 1
        S_int = self.R * T_int * rho_int ** (-gm1)          # 2
        R_p_int = V_n + 2 * a_int / gm1                     # 3

        # Interpolation values (assuming c = a_int). Transition smoothly on scale O(c)
        gating = self._gating(V_n, a_int)
        gating = torch.stack(gating, dim=-1)
        alpha1, alpha2, alpha3 = gating[:, 0], gating[:, 1], gating[:, 2]

        # Set boundary invariants
        R_m_bc = alpha1 * self.R_m_far + (1 - alpha1) * R_m_int
        S_bc = alpha2 * self.S_far + (1 - alpha2) * S_int
        R_p_bc = alpha3 * self.R_p_far + (1 - alpha3) * R_p_int

        # Reconstruct primatives
        V_n_bc = 1/2 * (R_m_bc + R_p_bc)
        a_bc_2 = (gm1/4 * (R_p_bc - R_m_bc)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_bc)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        # Add onto tangential component
        _, _, V_bc = self._recombine_Vs(V_n_bc, V_t)

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.bc_idx] = U_face_farfield
