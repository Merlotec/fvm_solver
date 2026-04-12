from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from torch import Tensor

from time_fvm.config_fvm import ConfigFVM, BCMode, ConfigBC
from time_fvm.sparse_utils import to_csr
if TYPE_CHECKING:
    from time_fvm.edge_process import FVMEdgeInfo
    from time_fvm.fvm_equation import PhysicalSetup


class BC:
    bc_normals: Tensor     # shape = [n_inlet_edges, 2]
    bc_tangents: Tensor

    def __init__(self, phy_setup: PhysicalSetup, cfg: ConfigFVM, bc_cfg: ConfigBC, bc_mask: Tensor, bc_normals: Tensor):
        """
        Generalised boundary condition.
        args:
            phy_setup: PhysicalSetup, contains EOS and other physics-specific functions.
            cfg: ConfigFVM, contains general configuration parameters.
            bc_cfg: ConfigBC, contains boundary condition specific parameters and mode.
            bc_mask: Tensor, shape [n_bc], which edges are set for this BC.
            bc_normals: Tensor, shape [n_bc], Outward pointing normals for boundary edges.
        """

        self.cfg = cfg
        self.device = cfg.device
        self.phy_setup = phy_setup

        # Boundary geometry
        self.bc_mask = bc_mask
        self.bc_normals = bc_normals
        self.bc_tangents = torch.stack((-bc_normals[:, 1], bc_normals[:, 0]), dim=1)
        self.bc_idx = torch.where(bc_mask)[0]

        # Shared boundary parameters
        self.T_inf = torch.tensor(bc_cfg.T_inf, device=self.device)
        self.rho_inf = torch.tensor(bc_cfg.rho_inf, device=self.device)
        self.v_n_inf = - torch.tensor(bc_cfg.v_n_inf, device=self.device)  # Since inlet edge points outwards
        self.v_t_inf = torch.tensor(bc_cfg.v_t_inf, device=self.device)
        p_inf = self.phy_setup.eos_P(self.rho_inf, self.T_inf)
        a_inf = self.phy_setup.eos_c(self.rho_inf, self.T_inf)

        # Set stagnation condition to be similar to natural parameters.
        match bc_cfg.mode:
            # Characteristic boundary condition
            case BCMode.Characteristic:
                self.set_bc_U_face = self.BC_characteristic

                self.p_inf = p_inf
                # We use this a lot
                self.ones = torch.ones(self.bc_mask.sum(), device=cfg.device)  # shape = [n_inlet]

            # Isentropic boundary condition
            case BCMode.Isentropic:
                self.set_bc_U_face = self.BC_isentropic

                self.gamma = phy_setup.gamma
                # Get stagnation conditions
                M_inf = self.v_n_inf / a_inf
                self.p_0 = p_inf * (1 + (self.gamma - 1) / 2 * M_inf ** 2) ** (self.gamma / (self.gamma - 1))
                self.T_0 = self.T_inf * (1 + (self.gamma - 1) / 2 * M_inf ** 2)

            # Farfield boundary condition
            case BCMode.Farfield:
                self.set_bc_U_face = self.BC_farfield

                self.R = phy_setup.R
                self.gamma = phy_setup.gamma
                self.R_far = self.v_n_inf - 2 * a_inf / (self.gamma - 1)

            # Farfield blended boundary condition
            case BCMode.FarfieldBlended:
                self.set_bc_U_face = self.BC_farfield_blended

                self.R = phy_setup.R
                self.gamma = phy_setup.gamma
                self.R_m_far = self.v_n_inf - 2 * a_inf / (self.gamma - 1)
                self.R_p_far = self.v_n_inf + 2 * a_inf / (self.gamma - 1)
                self.S_far = p_inf / (self.rho_inf ** self.gamma)

            case _ :
                raise NotImplementedError(f"Unknown Inlet Mode {bc_cfg.mode = }")

    def _construct_R(self, c: Tensor, rho: Tensor, ones=None):
        """ Construct R and R^-1 for the characteristic decomposition.
            A = [   [u, rho,        0       ],
                    [0, u,          1/rho   ],
                    [0, rho*c**2,   u       ]]
            Decompose A = R D R^-1

            c: shape = [n_bc]
            rho: shape = [n_bc]
            R: shape = [n_bc, 3, 3]
            R^-1: shape = [n_bc, 3, 3]
        """
        # Use cached ones matrix if needed.
        if ones is None:
            ones = self.ones

        zeros = torch.zeros_like(ones)
        c_sq = c ** 2

        R = torch.stack([
            torch.stack([ones, ones, ones], dim=1),
            torch.stack([-c / rho, zeros, c / rho], dim=1),
            torch.stack([c_sq, zeros, c_sq], dim=1),
        ], dim=1)  # shape = [n_inlet, 3, 3]

        R_inv = torch.stack([
            torch.stack([zeros, -rho / (2 * c), 1 / (2 * c_sq)], dim=1),
            torch.stack([ones, zeros, -1 / c_sq], dim=1),
            torch.stack([zeros, rho / (2 * c), 1 / (2 * c_sq)], dim=1),
        ], dim=1)

        return R, R_inv

    def _split_Vs(self, Vs):
        """ Split velocity into normal and tangential components.
            Vs: shape = [n_bc, 2]
            Return V_n, V_t: shape = [n_bc], [n_bc]
        """
        n = self.bc_normals
        t = self.bc_tangents

        v_n = (Vs * n).sum(dim=1)
        v_t = (Vs * t).sum(dim=1)

        return v_n, v_t

    def _recombine_Vs(self, v_n, v_t):
        """ Combine normal and tangential component of Vs back into x-y components."
            V_n: shape = [n_bc]
            V_t: shape = [n_bc]
            Return V_x, v_y: shape = [n_bc]
        """
        n = self.bc_normals
        t = self.bc_tangents

        V = v_n.unsqueeze(-1) * n + v_t.unsqueeze(-1) * t
        v_x, v_y = V[:, 0], V[:, 1]

        return v_x, v_y, V

    def _gating(self, v_n_int, c_int):
        """ Gating for forward and backward characteristics.
            v_n_int.shape = [n_bc]
            return.shape = [n_bc, 3]
        """
        lambda_vals = torch.stack([v_n_int - c_int, v_n_int, v_n_int + c_int], dim=-1)  # shape = [n_bc, 3]
        c = c_int.mean()
        lambda_scaled = (10/c) * lambda_vals                # shape = [n_bc, 3]
        gating = 0.5 * (1 - torch.tanh(lambda_scaled))

        return gating

    def set_bc_U_face(self, U_face, Us_bc_cells, dt):
        """ Set U_face.
            U_face: shape = [n_edges, n_comp], all boundary edges. Set value in place, given by mask.
            Us_bc_cells: shape = [n_bc_edges, n_comp], cell values at boundary edges.
        """
        raise NotImplementedError

    # ------------------------------- Specific BC implementations -------------------------------
    def BC_characteristic(self, U_face, Us_bc_cells, dt):
        """ Characteristic BC.
            Use W = (rho, v_n, p) -> dW/dt + div(f(W)) = 0
            Linearize using W = W_int + delta W
            Diagonalise equations
            Then solve for delta W_b by continuing characteristics from the left and right side.
            Tangential velocity is interpolated as well.

            U_face.shape = [n_bc_edges, n_comp], all boundary edges. Set value in place, given by mask.
            Us_bc_cells.shape = [n_inlet, n_comp], cell values at boundary edges.
        """
        # 1) Interior properties
        v_n_int, v_t_int = self._split_Vs(Us_bc_cells[:, [0, 1]])           # shape = [n_inlet]
        rho_int = Us_bc_cells[:, 2]  # shape = [n_inlet]
        T_int = Us_bc_cells[:, 3]  # shape = [n_inlet]
        # Convert basis to W = (rho, v_n, p)
        p_int = self.phy_setup.eos_P(rho_int, T_int)
        W_int = torch.stack([rho_int, v_n_int, p_int], dim=-1)  # shape = [n_inlet, 3]
        # Interior speed of sound
        c_int = self.phy_setup.eos_c(rho_int, T_int)

        # 2) Exterior properties
        rho_inf = self.rho_inf
        v_n_inf = self.v_n_inf
        p_inf = self.p_inf
        W_inf = torch.stack([rho_inf, v_n_inf, p_inf]) # shape = [3]

        # 3) Get transformation from dW basis to orthogonal dChi basis, decompose A = R D R^-1,
        R, R_inv = self._construct_R(c_int, rho_int)

        # 4) Compute dW and project into dChi space:
        dW_inf = W_inf - W_int
        # dChi = R^-1 dW to diagonalise the system into forward and backward components
        dChi_inf = (R_inv @ dW_inf.unsqueeze(-1)).squeeze()           # shape = [n_inlet, 3]

        # 5) Filter incoming and outgoing components. Transition smoothly on scale O(c)
        gating = self._gating(v_n_int, c_int)
        dChi_b = gating * dChi_inf        # shape = [n_inlet, 3]
        # 5.1) Tangential velocity is also interpolated using the gating
        v_t_int = gating[:, 1] * v_t_int + (1-gating[:, 1]) * self.v_t_inf

        # 6) Convert back to dW = R dChi,
        dW_b = (R @ dChi_b.unsqueeze(-1)).squeeze()                         # shape = [n_inlet, 3]
        W_b = W_int + dW_b

        # 7) Convert back to primatives
        rho_b = W_b[:, 0]
        # Keep tangential velocity from interior.
        v_n_b = W_b[:, 1]
        v_x_b, v_y_b, _ = self._recombine_Vs(v_n_b, v_t_int)
        # Convert pressure back into temperature
        p_b = W_b[:, 2]
        T_b = self.phy_setup.eos_T(rho_b, p_b)

        # Inplace update for U_face.
        U_face_b = torch.stack([v_x_b, v_y_b, rho_b, T_b], dim=-1)
        U_face[self.bc_mask] = U_face_b

    def BC_isentropic(self, U_face, Us_bc_cells, dt):
        """ Subsonic inlet boundary conditions based on isentropic flow relations (Adiabatic).
            Ideal gas only.
        Given stagnation (total) conditions p_0 and T_0, we compute the boundary state by combining interior flow.
        1. Pressure ratio (isentropic relation):
           p_0/p = (1 + (γ-1)/2 * M²)^(γ/(γ-1))
        2. Temperature ratio (isentropic relation):
           T_0/T = 1 + (γ-1)/2 * M²
        3. Speed of sound:
           a = √(γ * R * T)
        4. Mach number definition:
           M = V/a

        - p and T are computed from interior cell values, p=p_int, T=T_int
        - Backflow prevention: p_ratio is clamped to be ≥ 1 + ε to ensure M_bc ≥ 0
        - The tangential velocity component is preserved from the interior flow

        Parameters:
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_inlet_edges, n_comp]
            Interior cell values adjacent to inlet edges [V_x, V_y, ρ, T]
        dt : float
            Time step (unused, kept for interface compatibility)
        """

        V_int = Us_bc_cells[:, [0, 1]]  # shape = [n_inlet_edge, 2]
        rho_int = Us_bc_cells[:, 2]  # shape = [n_inlet_edge]
        T_int = Us_bc_cells[:, 3]  # shape = [n_inlet_edge]

        # Helper basis uses outward normals; inlet formulas below use inward normals.
        _, v_t_int = self._split_Vs(V_int)

        # Interior values
        p_int = self.phy_setup.eos_P(rho_int, T_int)
        p_ratio = self.p_0 / p_int
        p_ratio.clamp_(min=1 + 1e-7)  # Ensure no backflow.

        # Boundary values
        M_bc = torch.sqrt((2/(self.gamma - 1)) * (p_ratio ** ((self.gamma - 1)/self.gamma) - 1))
        T_bc = self.T_0 / (1 + (self.gamma - 1)/2 * M_bc ** 2)
        c_bc = self.phy_setup.eos_c(rho_int, T_bc)

        # Inlet velocity
        v_n_bc = - M_bc * c_bc
        v_x, v_y, _ = self._recombine_Vs(v_n_bc, v_t_int)

        rho_bc = self.phy_setup.eos_rho(p_int, T_bc)

        U_face_farfield = torch.stack([v_x, v_y, rho_bc, T_bc], dim=-1)
        U_face[self.bc_mask] = U_face_farfield

    def BC_farfield(self, U_face, Us_bc_cells, dt):
        """ Compressible farfield:
                R+ = u + 2a/(gamma - 1)
                R- = u - 2a/(gamma - 1)
                S = P / rho^gamma
            On exit, we have:
                R+ = R+_int
                R- = R-_inf
                S = S_int
        """
        gm1 = self.gamma - 1

        V = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]

        # Parallel and tangential velocity
        V_t, V_n = self._split_Vs(V)

        # Incoming farfield:
        R_m = self.R_far #self.v_far - 2 * self.a_far / gm1

        # Outgoing (extrapolate from internal) :
        a_int = torch.sqrt(self.gamma * self.R * T_int)
        R_p = V_n + 2 * a_int / gm1       #  R+ = R+_int = V_n + 2 * a_in / (gamma-1)
        S_p = self.R * T_int * rho_int ** (-gm1)

        # Boundary values: a_bc = (gamma-1)/4 * (R+ - R-)
        V_n_bc = 1/2 * (R_p + R_m)
        a_bc_2 = (gm1/4 * (R_p - R_m)) ** 2
        rho_bc = (a_bc_2 / (self.gamma * S_p)) ** (1 / gm1)
        T_bc = a_bc_2 / (self.gamma * self.R)

        _, _, V_bc = self._recombine_Vs(V_n_bc, V_t)

        U_face_farfield = torch.cat([V_bc, rho_bc.unsqueeze(-1), T_bc.unsqueeze(-1)], dim=-1)
        U_face[self.bc_mask] = U_face_farfield

    def BC_farfield_blended(self, U_face, Us_bc_cells, dt):
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
        U_face : torch.Tensor, shape [n_edges_bc, n_comp]
            Boundary face values to be modified in place
        Us_bc_cells : torch.Tensor, shape [n_farfield_edges, n_comp]
            Interior cell values adjacent to farfield edges [V_x, V_y, ρ, T]
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
        V_int = Us_bc_cells[:, [0, 1]]                                    # shape = [n_ff_edge, 2]
        rho_int = Us_bc_cells[:, 2]                                   # shape = [n_ff_edge]
        T_int = Us_bc_cells[:, 3]                                     # shape = [n_ff_edge]
        # Parallel and tangential velocity
        V_n, V_t = self._split_Vs(V_int)

        # Interior invariants:
        a_int = self.phy_setup.eos_c(rho_int, T_int)
        R_m_int = V_n - 2 * a_int / gm1                     # 1
        S_int = self.R * T_int * rho_int ** (-gm1)          # 2
        R_p_int = V_n + 2 * a_int / gm1                     # 3

        # Interpolation values (assuming c = a_int). Transition smoothly on scale O(c)
        gating = self._gating(V_n, a_int)
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
        U_face[self.bc_mask] = U_face_farfield


class BoundarySetter:
    """ Non-orthogonal correction for Neumann BCs."""
    n_comp: int
    n_edges_bc: int

    grad_comps: torch.Tensor          # shape = [n_neum_edges, 2, 1]
    where_neum: tuple[torch.Tensor]      # shape = [2][n_neum_edges, 2]

    # Matrices for general face values
    A_bc: torch.Tensor
    b_bc: torch.Tensor

    # Farfield boundary condition
    use_farfield: bool
    farfield_calc: BC
    exit_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Exit edge for each cell

    # Inlet boundary condition
    use_inlet: bool
    inlet_calc: BC
    inlet_cell2edge: torch.Tensor    # shape = (n_cells, 2)  # Inlet edge for each cell

    def __init__(self, E_props: FVMEdgeInfo, phy_setup: PhysicalSetup):
        self.phy_setup = phy_setup

        self.use_farfield, self.use_inlet = False, False

        self.n_comp = E_props.n_comp
        self.n_edges_bc = E_props.n_edges_bc
        tri_to_edge = E_props.tri_to_edge.view(-1, 3)

        # Flatten out all Neumann BCs and index according to order where_neum_all[0]
        neum_mask_all = torch.zeros_like(E_props.bc_edge_mask)
        neum_mask_all = neum_mask_all.unsqueeze(-1).repeat(1, 4)
        neum_mask_all[E_props.bc_edge_mask] = E_props.neumann_mask
        where_neum_all = torch.where(neum_mask_all)
        where_neum = {'edge': where_neum_all[0], 'comp': where_neum_all[1]}  # shape = [n_neum_edges, 2]
        # Mapping from boundary id to boundary edge id
        self.where_neum = torch.where(E_props.neumann_mask)

        # Mapping from boundary edge to cell
        bc_edge_to_tri = torch.zeros_like(E_props.bc_edge_mask).long()
        bc_edge_to_tri[E_props.bc_edge_mask] = E_props.edge_to_tri_bc

        # Cells corresponding to Neumann BC
        self.neum_cells = bc_edge_to_tri[where_neum_all[0]]  # shape = [n_neum_edges], which cells have neuman BCs
        where_neum['cells'] = self.neum_cells
        # Edge within cell corresponding to Neumann BC
        tri_edge_num = (where_neum['edge'].unsqueeze(-1).repeat(1, 3) == tri_to_edge[where_neum['cells']])
        tri_edge_id = torch.where(tri_edge_num)[1]
        where_neum['tri_edge_id'] = tri_edge_id

        # Which component of gradient is needed for Neumann BC
        self.grad_comps = where_neum['comp'].unsqueeze(1).repeat(1, 2).unsqueeze(2)     # shape = [n_neum_edges, 2, 1]
        # Normal vector of edges
        n_hats = E_props.normals_hat.squeeze()[where_neum['edge']]  # shape = [n_neum_edges, 2]
        # Displacement from centroid to edge
        cent_to_edge = E_props.cent_to_edge_disp[where_neum['cells']].squeeze()  # shape = [n_neum_edge, 3, 2]
        r = cent_to_edge[torch.arange(cent_to_edge.shape[0]), where_neum['tri_edge_id']]  # shape = [n_neum_edge, 2]
        # Normal component of r
        d = n_hats * (r * n_hats).sum(dim=1, keepdim=True)  # shape = [n_neum_edge, 2]
        # Parallel component of r
        self.l = r - d

        A_bc, b_bc = self._build_spm_face_vals(E_props)
        self.A_bc, self.b_bc = A_bc, b_bc

    def set_face_values(self, Us, cell_grads=None, dt=None):
        """Compute and return boundary face values from cell values.

        Uses the precomputed sparse operator to map flattened cell values to
        boundary face values, then applies non-orthogonal correction and
        farfield adjustments if enabled.
        """
        Us_flat = Us.flatten()
        # Final U_face in flattened form.a
        U_face_flat = torch.mv(self.A_bc, Us_flat) + self.b_bc      # shape = [n_edges_bc * n_comp]
        # Reshape back to (n_edges_bc, n_comp)
        U_face = U_face_flat.view(self.n_edges_bc, self.n_comp)

        if cell_grads is not None:
            self._non_orthogonal_correction(U_face, cell_grads)

        if self.use_farfield:
            self.farfield_calc.set_bc_U_face(U_face, Us[self.exit_cell2edge], dt)

        if self.use_inlet:
            self.inlet_calc.set_bc_U_face(U_face, Us[self.inlet_cell2edge], dt)

        return U_face

    def _non_orthogonal_correction(self, U_face, cell_grads):
        """
        Use previous gradient for non-orthogonal correction.

        U_face.shape = [n_bc_faces, n_comp]
        cell_grads.shape = [n_cells, 2, n_comp]

        r = centroid to midpoint.
        d = normal component of r
        U_f = U_0 + d * dUdn + (r-d) grad(U)
        """
        grads = torch.gather(cell_grads[self.neum_cells], 2, self.grad_comps).squeeze()  # shape = [n_neum_edge, 2]

        dU = (grads * self.l).sum(dim=1)  # shape = [n_neum_edge]
        U_face[self.where_neum[0], self.where_neum[1]] += dU

    def _build_spm_face_vals(self, E_props: FVMEdgeInfo):
        """ Compute bc edge values using sparse matrix multiplication. """

        device = E_props.device
        n_bc = E_props.n_edges_bc  # number of boundary edges
        n_comp = E_props.n_comp
        n_cells = E_props.n_cells

        # Total number of flattened BC rows.
        N = n_bc * n_comp

        # Create flattened indices for the boundary rows and the corresponding component.
        # Each boundary edge gives rise to n_comp rows.
        bc_rows = torch.arange(n_bc, device=device).unsqueeze(1).expand(n_bc, n_comp).reshape(-1)
        comp_idx = torch.arange(n_comp, device=device).unsqueeze(0).expand(n_bc, n_comp).reshape(-1)

        # Reshape the condition masks to a flat vector of length N.
        dirich_mask = E_props.dirich_mask.reshape(-1)  # For Dirichlet conditions.
        neum_mask = E_props.neumann_mask.reshape(-1)  # For Neumann conditions.

        # --- Build sparse matrix A ---
        # For Neumann entries, we want to extract the cell value from Us.
        # For each Neumann row, the corresponding column in Us (flattened) is given by:
        #   col = self.edge_to_tri_bc[ edge_index ] * n_comp + component
        neum_indices = torch.nonzero(neum_mask, as_tuple=False).squeeze(1)  # indices where Neumann is True.
        A_rows = neum_indices
        # bc_rows[neum_indices] gives the corresponding boundary edge for each flattened row.
        A_cols = E_props.edge_to_tri_bc[bc_rows[neum_indices]] * n_comp + comp_idx[neum_indices]
        A_vals = torch.ones_like(A_rows, dtype=torch.float32, device=device)

        size_A = (N, n_cells * n_comp)
        A_bc = torch.sparse_coo_tensor(torch.stack([A_rows, A_cols], dim=0), A_vals, size=size_A)# .coalesce().to_sparse_csr()
        A_bc = to_csr(A_bc, device=device)
        # Build the offset vector b.
        b_bc = torch.empty(N, device=device, dtype=torch.float32)
        # For Dirichlet entries, the prescribed value should override any extracted value.
        b_bc[dirich_mask] = E_props.dirich_val
        # For Neumann entries, add the offset computed from the edge distance.
        # Here, we select the proper component value from self.neumann_val using comp_idx.
        b_bc[neum_mask] = E_props.neumann_val[comp_idx[neum_mask]] * E_props.edge_dists_bc.flatten()[neum_mask]

        return A_bc, b_bc

    def init_farfield(self, cfg: ConfigFVM, farfield_mask, exit_cell2edge, farfield_normals):
        self.use_farfield = True
        self.farfield_calc = BC(self.phy_setup, cfg, cfg.exit_cfg, farfield_mask, farfield_normals)
        self.exit_cell2edge = exit_cell2edge

    def init_inlet(self, cfg: ConfigFVM, inlet_mask, inlet_cell2edge, inlet_normals):
        self.use_inlet = True
        self.inlet_calc = BC(self.phy_setup, cfg, cfg.inlet_cfg, inlet_mask, inlet_normals)
        self.inlet_cell2edge = inlet_cell2edge



