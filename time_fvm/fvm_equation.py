from typing import TYPE_CHECKING
import torch
from abc import ABC
from cprint import c_print

from time_fvm.utils.plotting import plot_points, plot_interp_cell, plot_edges
from time_fvm.fvm_stepping.facet_process import FacetFlux
from time_fvm.time_solvers.t_solvers import FVMCells
from time_fvm.time_solvers.integrators import get_solver

if TYPE_CHECKING:
    from time_fvm.mesh_utils.fvm_mesh import FVMMesh
    from config_fvm import ConfigFVM


class PhysicalSetup:
    """ Set physical properties of fluid. """
    dim: int
    tau: torch.Tensor       # shape = [n_edges, 2, 2]
    P_face: torch.Tensor    # shape = [n_edges, 2, 1]
    c: torch.Tensor         # shape = [n_edges, 2, 1]

    def __init__(self, cfg: ConfigFVM, dim: int):
        self.device = cfg.device
        self.dim = dim

        self.T_0 = cfg.T_0
        self.mu = cfg.viscosity
        self.mu_b = cfg.visc_bulk
        self.S_const = cfg.S_const
        self.C_v = cfg.C_v
        self.gamma = cfg.gamma
        self.R = self.C_v * (cfg.gamma - 1)

    def state_to_primative(self, state: torch.Tensor):
        """ Convert from conserved quantities (momentum, rho, energy) to primitives (velocity, rho, T) """
        d = self.dim
        momentum, rho, Q = state[:, :d], state[:, d:d+1], state[:, d+1:d+2]

        V = momentum / rho
        T = 1 / self.C_v * (Q / rho - 0.5 * V.square().sum(dim=1, keepdim=True))
        primatives = torch.cat([V, rho, T], dim=-1)

        return primatives, state

    def primatives_to_state(self, V, rho, T):
        """ Convert from primitives (velocity, rho, T) to conserved quantities (momentum, rho, energy)
            V.shape = [..., dim]
            rho.shape = [..., 1]
            T.shape = [..., 1]
         """

        momentum = V * rho
        Q = rho * (self.C_v * T + 0.5 * V.square().sum(dim=-1, keepdim=True))
        return momentum, rho, Q

    def _tau(self, E_props: FacetFlux):
        """ Compute stress tensor:
                tau = mu * (grad(V) + grad(V).T) + mu_b * div(V) * I
         """
        T = E_props.T_facet       # shape = [n_edges, edges=2, n_comp=1]

        # Strain and invariants
        D, I1, I2 = self._strain_values(E_props)        # shape = [n_edges, 2, 2]
        I1 = I1.view(-1, 1, 1)   # [n_edges, 1, 1]. Trace of D.

        # Viscosity = mu * (T/T0)^(3/2) * (T0 + S) / (T + S)
        mu = self.mu * (T /  self.T_0)**1.5 * (self.T_0 + self.S_const) / (T + self.S_const)  # shape = [n_edges, edges=2, n_comp=1]
        # Bulk viscosity: Proportional to T^2
        mu_b = self.mu_b * T ** 2 / self.T_0 ** 2

        eye = torch.eye(2, device=self.device)
        self.tau = -2 * mu * D - mu_b * I1 * eye  # shape = [n_edges, 2, 2]

    def _strain_values(self, E_props: FacetFlux):
        """ Compute strain tensor:
                epsilon = 0.5 * (grad(V) + grad(V).T)
            Then compute the 2D invariants:
                I1 = tr(D)  (divergence)
                I2 = tr(D^2) (Magnitude of deformation)
         """
        grad_V_t = E_props.grad_V  # shape = [n_edges, dim=2, n_comp=2]
        D = 0.5 * (grad_V_t + grad_V_t.permute(0, 2, 1))  # shape = [n_edges, 2, 2]

        # Invariants
        I1 = D[:, 0, 0] + D[:, 1, 1]  # shape = [n_edges]
        I2 = (D**2).sum(dim=(-1, -2))       # Since D is symmetric, this is faster.

        return D, I1, I2

    def _pressure(self, E_props: FacetFlux):
        """ Pressure force:
                P = rho * C_v * (gamma - 1) * T = R * rho * T
        """
        rho_faces = E_props.rho_facet               # shape = [n_edges, edges=2, n_comp=1]
        T_faces = E_props.T_facet                   # shape = [n_edges, edges=2, n_comp=1]

        self.P_face = self.eos_P(rho_faces, T_faces)
        self.c = self.eos_c(rho_faces, T_faces)     # shape = [n_edges, edges=2, n_comp=1]
        # self.c, self.P_face = self.eos_cP(rho_faces, T_faces)

    # General gas parameters.
    def eos_c(self, rho, T):
        """ Speed of sound, c^2 = dp/drho | s"""
        return torch.sqrt(self.gamma * self.R * T)

    def eos_P(self, rho, T):
        """ Pressure EOS """
        return self.R * rho * T

    def eos_T(self, rho, P):
        """ Inverse of eos_P """
        return P / (self.R * rho)

    def update(self, E_props: FacetFlux):
        self._tau(E_props)
        self._pressure(E_props)


class FVMEdgeFunc(ABC):
    device: str

    #@abstractmethod
    def edge_fluxes(self, fluxes=None):
        """ Compute flux for each edge """
        pass


class Adevction(FVMEdgeFunc):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
        dims: Which dimensions of Us are advected.
    """
    E_props: FacetFlux

    def __init__(self, E_props: FacetFlux, phy_setup: PhysicalSetup, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup

    def edge_fluxes(self, fluxes=None):
        """ rho * U @ V.T @ n = rho V * phi
            f_x = rho V_x * phi
            f_y = rho V_y * phi
            f_rho = rho * phi
            f_E = (Q+p) * phi
        """
        E_props = self.E_props
        rho_faces = E_props.rho_facet # shape = [n_edges, edges=2, n_comp=1]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_edges, edges=2]
        mom_f = E_props.mom_facet
        Q_faces = E_props.Q_facet

        Q_p_P = Q_faces + self.phy_setup.P_face
        Us_f = torch.cat([mom_f, rho_faces, Q_p_P], dim=-1)  # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = Us_f * phi.unsqueeze(-1)           # shape = [n_edges, edges=2, n_comp=3]
        advec_flux = advec_flux.mean(dim=1)              # shape = [n_edges, n_comp=3]

        if fluxes is None:
            return advec_flux.contiguous()
        else:
            fluxes += advec_flux


class Viscosity(FVMEdgeFunc):
    """ Viscous forces:
            Shear viscosity: div(mu grad(V)) = sum_f grad(V) * mu_f * l_f
            Bulk viscosity: k * grad(div(V))
    """
    E_props: FacetFlux
    def __init__(self, E_props: FacetFlux, stress_calc: PhysicalSetup, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.mesh = E_props.mesh
        self.stress_calc = stress_calc

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props

        tau = self.stress_calc.tau
        F = (tau * self.mesh.normals.unsqueeze(-1)).sum(dim=-2)  # shape = [n_facets, 2]

        if fluxes is None:
            fluxes = torch.zeros(E_props.n_facets, E_props.n_comp, device=self.device)
            fluxes[:, :2] = F
            return fluxes
        else:
            fluxes[:, :2] += F


class Heating(FVMEdgeFunc):
    """ Viscous heating term: div(tau V) = sum_f tau_f * V_f * n_f
        Thermal conductivity term:  div(grad(T)) = sum(grad(T) * n_f)
    """
    E_props: FacetFlux
    def __init__(self, E_props: FacetFlux, stress_calc:PhysicalSetup, cfg: ConfigFVM, device="cpu"):
        self.E_props = E_props
        self.stress_calc = stress_calc

        self.kappa = cfg.thermal_cond
        self.device = device

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props
        mesh = E_props.mesh
        normals = mesh.normals          # shape = [n_facets, 2]
        V_face = E_props.Vs_facet       # shape = [n_facets, facets=2, n_comp=2]

        tau = self.stress_calc.tau      # shape = [n_facets, 2, 2]

        heating = (tau * V_face * normals.unsqueeze(-1)).sum(dim=(-1, -2))      # Take average over all faces.

        """ Thermal conductivity:
                div(grad(T)) = sum(grad(T) * n_f)
        """
        grad_T_n = E_props.grad_T_n     # shape = [n_facets]
        # heating -= self.kappa * grad_T_n * mesh.edge_len.squeeze()
        heating.addcmul_(grad_T_n, mesh.facet_size.squeeze(), value=-self.kappa)

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_facets, self.E_props.n_comp, device=self.device)
            fluxes[:, 3] = heating
            return fluxes
        else:
            fluxes[:, 3] += heating


class PressureForce(FVMEdgeFunc):
    """ Special case. N
        grad(rho) = div(rho I) """
    E_props: FacetFlux

    def __init__(self, E_props: FacetFlux, phy_setup: PhysicalSetup, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup

    def edge_fluxes(self, fluxes=None):
        normals = self.E_props.mesh.normals             # shape = [n_facets, 2]
        P_face = self.phy_setup.P_face      # shape = [n_facets, facets=2, n_comp=1]

        P_face = P_face.mean(dim=1)  # shape = [n_facets, 1]
        P_n = P_face * normals                 # shape = [n_facets, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_facets, self.E_props.n_comp, device=self.device)
            fluxes[:, :2] = P_n
            return fluxes # _flat
        else:
            fluxes[:, :2] += P_n


class KTDiffusion(FVMEdgeFunc):
    """ Diffusion term from K-T solver.
        Flux = a/2 * (U_L - U_R) * edge_len
     """
    E_props: FacetFlux

    def __init__(self, E_props: FacetFlux, v_factor, phy_setup: PhysicalSetup, device="cpu"):
        self.device = device
        self.v_factor = v_factor
        self.E_props = E_props
        self.phy_setup = phy_setup
        self.a_clip = 1

    def edge_fluxes(self, fluxes=None):
        E_props = self.E_props
        rho_face = E_props.rho_facet
        Vs_face = E_props.Vs_facet      # shape = [n_edges, edges=2, n_comp=2]
        Q_face = E_props.Q_facet   # shape = [n_edges, edges=2, n_comp=1]
        mom_face = E_props.mom_facet
        edge_len = E_props.mesh.facet_size

        # Us = torch.cat([mom_face, rho_face, Q_face], dim=2)  # shape = [n_edges, 2, n_comp]

        # Wavespeed is c + v_max. Clip velocity wavespeed to k*c + v_max
        Vs = Vs_face.norm(dim=-1)            # shape = [n_edges, edges=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values   # shape = [n_edges, 1]
        c = self.phy_setup.c.max(dim=1).values  # shape = [n_edges, 1]

        # Reduce diffusion for velocity for low Mach number flows
        M = (Vs_max / (c + 1e-8)).abs()
        v_factor = torch.clamp(M, min=self.v_factor, max=1)

        # Use different diffusion for velocity and other components. For velocity, use v_factor * c, for others use c.
        a_vel = v_factor * c + Vs_max   # shape = [n_edges, n_comp]
        a_other = c + Vs_max            # shape = [n_edges, n_comp]

        # Flux = a/2 * (U_L - U_R) * edge_len
        dmom = mom_face[:, 0] - mom_face[:, 1]  # [n_edges, 2]
        drho = rho_face[:, 0] - rho_face[:, 1]  # [n_edges, 1]
        dQ = Q_face[:, 0] - Q_face[:, 1]  # [n_edges, 1]

        kt_fluxes = 0.5 * torch.cat([a_vel * dmom, a_other * drho, a_other * dQ], dim=1) * edge_len

        if fluxes is None:
            return kt_fluxes
        else:
            fluxes += kt_fluxes


class FVMEquation:
    mesh: FVMMesh
    E_props: FacetFlux
    edges: FVMEdgeFunc
    cells: FVMCells
    n_comp: int

    def __init__(self, cfg: ConfigFVM, phy_setup: PhysicalSetup, mesh: FVMMesh, n_comp, bc_tag, us_init=None):
        self.cfg = cfg
        self.phy_setup = phy_setup
        self.device = cfg.device
        self.mesh = mesh
        self.n_comp = n_comp

        device = self.device

        # Physical parameters

        E_props = FacetFlux(self.phy_setup, cfg, mesh, n_comp, bc_tag, device=device)
        self.cells = FVMCells(mesh.n_cells, n_comp, init_val=us_init, phys_setup=self.phy_setup, device=device)
        self.flux_mat = E_props.mesh.flux_mat
        self.E_props = E_props

        self.P_force = PressureForce(E_props, self.phy_setup, device=device)
        self.U_advect = Adevction(E_props, self.phy_setup, device=device)
        self.U_visc = Viscosity(E_props, self.phy_setup, device=device)
        self.Heat = Heating(E_props, self.phy_setup, cfg=cfg, device=device)
        self.KT_diff = KTDiffusion(E_props, cfg.v_factor, self.phy_setup, device=device)

        self.t_solver = get_solver(self.cells, self, cfg)

        E_props.clear_temp()
        c_print("Done FVMEquation", color="bright_magenta")

    def solve(self):
        self.t_solver.solve()

    def forward(self, primatives):
        """ primatives.shape = (n_cells, n_comp) """
        E_props = self.E_props
        E_props.compute_facet(primatives)

        self.phy_setup.update(E_props)

        # Advection term
        fluxes = self.U_advect.edge_fluxes()
        # Pressure term
        self.P_force.edge_fluxes(fluxes)
        # Viscosity term
        self.U_visc.edge_fluxes(fluxes)
        # Heating term
        self.Heat.edge_fluxes(fluxes)
        # MUSCL term
        self.KT_diff.edge_fluxes(fluxes)
        # Compute divergence
        divergence = self._flux_to_div(fluxes)

        return divergence

    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges * N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        divergence = self.flux_mat.spMM(fluxes)  # shape: (n_cells * n_component,)
        return divergence

    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None, Xlims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.facets.cpu(), title=title, colors=fluxes, show_index=show_index, lims=lims, Xlims=Xlims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None, Xlims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims, Xlims=Xlims)

    def plot_interp(self, values, title="Cell Values", Xlims=None, resolution=2000):
        plot_interp_cell(self.mesh.vertices, values.T, self.mesh.cells, title=title, Xlims=Xlims)

    def pretty_plot(self, primatives, Xlims=None, title=None):
        Vx, Vy, rho, T = primatives[:, 0], primatives[:, 1], primatives[:, 2], primatives[:, 3]

        P = self.phy_setup.R * rho * T
        c = torch.sqrt(P / rho)

        Mx, My = Vx / c, Vy / c
        M_num = torch.sqrt(Mx ** 2 + My ** 2)

        plot_vals = torch.stack([P, M_num, self.divergence[:, 3] ], dim=0)

        title = [f"Pressure: {title}", f"Mach number: {title}", f'Heating: {title}']
        plot_interp_cell(self.mesh.vertices, plot_vals, self.mesh.cells, title=title, Xlims=Xlims)
