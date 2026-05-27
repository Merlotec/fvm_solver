from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from abc import ABC, abstractmethod
from cprint import c_print

from time_fvm.utils.plotting import plot_points, plot_interp_cell, plot_edges
from time_fvm.fvm_stepping.facet_process import FacetCalc
from time_fvm.time_solvers.t_solvers import FVMCells
from time_fvm.time_solvers.integrators import get_solver
if TYPE_CHECKING:
    from time_fvm.mesh_utils.fvm_mesh import FVMMesh
    from config_fvm import ConfigFVM


class FluidConstitution(ABC):
    """ Set physical properties of fluid. """
    dim: int
    tau: torch.Tensor       # shape = [n_facets, 2, dim]
    P_face: torch.Tensor    # shape = [n_facets, 2, 1]
    c: torch.Tensor         # shape = [n_facets, 2, 1]

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

    @abstractmethod
    def _tau(self, E_props: FacetCalc):
        """ Compute stress tensor tau """
        pass

    def _pressure(self, E_props: FacetCalc):
        """ Pressure force:
                P = rho * C_v * (gamma - 1) * T = R * rho * T
        """
        rho_faces = E_props.rho_facet               # shape = [n_facets, facets=2, n_comp=1]
        T_faces = E_props.T_facet                   # shape = [n_facets, facets=2, n_comp=1]

        self.P_face = self.eos_P(rho_faces, T_faces)
        self.c = self.eos_c(rho_faces, T_faces)     # shape = [n_facets, facets=2, n_comp=1]

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

    def update(self, E_props: FacetCalc):
        self._tau(E_props)
        self._pressure(E_props)


class FluidConstitution2D(FluidConstitution):
    """ Set physical properties of fluid. """
    dim: int
    tau: torch.Tensor       # shape = [n_facets, 2, 2]
    P_face: torch.Tensor    # shape = [n_facets, 2, 1]
    c: torch.Tensor         # shape = [n_facets, 2, 1]

    def __init__(self, cfg: ConfigFVM, dim: int):
        super().__init__(cfg, dim)
        assert self.dim == 2, "Only for 2D fluids"

    def _tau(self, E_props: FacetCalc):
        """ Compute stress tensor:
                tau = mu * (grad(V) + grad(V).T) + mu_b * div(V) * I
         """
        T = E_props.T_facet.mean(dim=1).squeeze()         # shape = [n_facets]
        grad_V_t = E_props.grad_V   # shape = [n_facets, dim=2, n_comp=2]

        # Strain and invariants
        D, I1, I2 = self._strain_values(grad_V_t)        # shape = [n_facets, 2, 2]

        # Viscosity = mu * (T/T0)^(3/2) * (T0 + S) / (T + S)
        mu = self.mu * (T /  self.T_0)**1.5 * (self.T_0 + self.S_const) / (T + self.S_const)  # shape = [n_facets]
        # Bulk viscosity: Proportional to T^2
        mu_b = self.mu_b * T ** 2 / self.T_0 ** 2

        a0 = (-mu_b * I1).view(-1, 1, 1)         # shape = [n_facets, 1, 1]
        a1 = (-2 * mu).view(-1, 1, 1)

        eye = torch.eye(2, device=self.device).unsqueeze(0)
        self.tau = a0 * eye + a1 * D  # shape = [n_facets, 2, 2]

    def _strain_values(self, grad_V_t: torch.Tensor):
        """ Compute strain tensor:
                epsilon = 0.5 * (grad(V) + grad(V).T)
            Then compute the 2D invariants:
                I1 = tr(D)  (divergence)
                I2 = tr(D^2) (Magnitude of deformation)
         """
        D = 0.5 * (grad_V_t + grad_V_t.permute(0, 2, 1))    # shape = [n_facets, 2, 2]

        # Invariants
        I1 = D[:, 0, 0] + D[:, 1, 1]                        # shape = [n_facets]
        I2 = (D**2).sum(dim=(-1, -2))       # Since D is symmetric, this is faster.

        return D, I1, I2


class FluidConstitution3D(FluidConstitution):
    """ Set physical properties of fluid. """
    dim: int
    tau: torch.Tensor  # shape = [n_facets, 2, 3]
    P_face: torch.Tensor  # shape = [n_facets, 2, 1]
    c: torch.Tensor  # shape = [n_facets, 2, 1]

    def __init__(self, cfg: ConfigFVM, dim: int):
        super().__init__(cfg, dim)
        assert self.dim == 3, "Only for 3D fluids"

    def _tau(self, E_props: FacetCalc):
        """ Compute stress tensor:
                I1, I2, I3 = tr(D), tr(D^2), tr(D^3)
                tau = a0 I + a1 D + a2 D^2
         """
        T = E_props.T_facet.mean(dim=1).squeeze()         # shape = [n_facets]
        grad_V_t = E_props.grad_V   # shape = [n_facets, 3, 3]

        # Strain and invariants
        D, D2, I1, I2, I3 = self._strain_values(grad_V_t)    # shape = [n_facets, 3, 3]

        # Viscosity = mu * (T/T0)^(3/2) * (T0 + S) / (T + S)
        mu = self.mu * (T / self.T_0) ** 1.5 * (self.T_0 + self.S_const) / (T + self.S_const)  # shape = [n_facets]
        # Bulk viscosity: Proportional to T^2
        mu_b = self.mu_b * T ** 2 / self.T_0 ** 2

        a0 = (-mu_b * I1).view(-1, 1, 1)
        a1 = (-2 * mu).view(-1, 1, 1)
        a2 = 0

        eye = torch.eye(3, device=self.device)
        self.tau = a0 * eye + a1 * D  + a2 * D2 # shape = [n_facets, 3, 3]

    def _strain_values(self, grad_V_t: torch.Tensor):
        """ Compute strain tensor:
                epsilon = 0.5 * (grad(V) + grad(V).T)
            Then compute the 2D invariants:
                I1 = tr(D)  (divergence)
                I2 = tr(D^2) (Magnitude of deformation)
                I3 = tr(D^3) (Asymmetry of deformation)
         """
        D = 0.5 * (grad_V_t + grad_V_t.permute(0, 2, 1))  # shape = [n_facets, 3, 3]
        D2 = D @ D

        # Invariants
        I1 = D[:, 0, 0] + D[:, 1, 1]  # shape = [n_facets]
        I2 = D2[:, 0, 0] + D2[:, 1, 1]
        I3 = torch.einsum("...ij,...jk,...ki->...", D, D, D)
        return D, D2, I1, I2, I3


class FVMFacetFlux(ABC):
    device: str

    #@abstractmethod
    def facet_fluxes(self, fluxes=None):
        """ Compute flux for each facet """
        pass


class Adevction(FVMFacetFlux):
    """ out_i = div(rho V V_i) for velocity V, i = {x, y}
        dims: Which dimensions of Us are advected.
    """
    E_props: FacetCalc

    def __init__(self, E_props: FacetCalc, phy_setup: FluidConstitution, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup

    def facet_fluxes(self, fluxes=None):
        """ rho * U @ V.T @ n = rho V * phi
            f_x = rho V_x * phi
            f_y = rho V_y * phi
            f_rho = rho * phi
            f_E = (Q+p) * phi       (compressional heating)
        """
        E_props = self.E_props
        rho_faces = E_props.rho_facet # shape = [n_facets, facets=2, n_comp=1]
        phi = E_props.phi           # Linear interpolation of convection vector = (v_faces dot normal). shape = [n_facets, facets=2]
        mom_f = E_props.mom_facet
        Q_faces = E_props.Q_facet

        Q_p_P = Q_faces + self.phy_setup.P_face
        Us_f = torch.cat([mom_f, rho_faces, Q_p_P], dim=-1)  # shape = [n_facets, facets=2, n_comp=3]
        advec_flux = Us_f * phi.unsqueeze(-1)           # shape = [n_facets, facets=2, n_comp=3]
        advec_flux = advec_flux.mean(dim=1)              # shape = [n_facets, n_comp=3]

        if fluxes is None:
            return advec_flux.contiguous()
        else:
            fluxes += advec_flux


class Viscosity(FVMFacetFlux):
    """ Viscous forces:
            div(mu grad(V)) = sum_f tau_f * n_f
    """
    E_props: FacetCalc
    def __init__(self, E_props: FacetCalc, stress_calc: FluidConstitution, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.mesh = E_props.mesh
        self.stress_calc = stress_calc
        self.dim = E_props.dim

    def facet_fluxes(self, fluxes=None):
        E_props = self.E_props

        tau = self.stress_calc.tau
        F = (tau * self.mesh.normals.unsqueeze(-1)).sum(dim=-2)  # shape = [n_facets, dim]

        if fluxes is None:
            fluxes = torch.zeros(E_props.n_facets, E_props.n_comp, device=self.device)
            fluxes[:, :self.dim] = F
            return fluxes
        else:
            fluxes[:, :self.dim] += F


class Heating(FVMFacetFlux):
    """ Viscous heating term: div(tau V) = sum_f tau_f * V_f * n_f
        Thermal conductivity term:  div(grad(T)) = sum(grad(T) * n_f)
    """
    E_props: FacetCalc
    def __init__(self, E_props: FacetCalc, stress_calc:FluidConstitution, cfg: ConfigFVM, device="cpu"):
        self.E_props = E_props
        self.stress_calc = stress_calc
        self.dim = E_props.dim
        self.kappa = cfg.thermal_cond
        self.device = device

    def facet_fluxes(self, fluxes=None):
        E_props = self.E_props
        mesh = E_props.mesh
        normals = mesh.normals                                                  # shape = [n_facets, dim]
        V_face = E_props.Vs_facet                                               # shape = [n_facets, facets=2, n_comp=dim]

        tau = self.stress_calc.tau                                              # shape = [n_facets, dim, dim]
        V_face = V_face.mean(dim=1, keepdim=True)                               # shape = [n_facets, 1, dim]
        heating = (tau * V_face * normals.unsqueeze(-1)).sum(dim=(-1, -2))      # shape = [n_facets], n.Tau.V

        """ Thermal conductivity:
                div(grad(T)) = sum(grad(T) * n_f)
        """
        grad_T_n = E_props.grad_T_n     # shape = [n_facets]
        # heating -= self.kappa * grad_T_n * mesh.facet_len.squeeze()
        heating.addcmul_(grad_T_n, mesh.facet_size.squeeze(), value=-self.kappa)

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_facets, self.E_props.n_comp, device=self.device)
            fluxes[:, self.dim+1] = heating
            return fluxes
        else:
            fluxes[:, self.dim+1] += heating


class PressureForce(FVMFacetFlux):
    """ Special case. N
        grad(rho) = div(rho I) """
    E_props: FacetCalc
    dim: int

    def __init__(self, E_props: FacetCalc, phy_setup: FluidConstitution, device="cpu"):
        self.device = device
        self.E_props = E_props
        self.phy_setup = phy_setup
        self.dim = phy_setup.dim

    def facet_fluxes(self, fluxes=None):
        normals = self.E_props.mesh.normals             # shape = [n_facets, 2]
        P_face = self.phy_setup.P_face      # shape = [n_facets, facets=2, n_comp=1]

        P_face = P_face.mean(dim=1)  # shape = [n_facets, 1]
        P_n = P_face * normals                 # shape = [n_facets, 2]

        if fluxes is None:
            fluxes = torch.zeros(self.E_props.n_facets, self.E_props.n_comp, device=self.device)
            fluxes[:, :self.dim] = P_n
            return fluxes # _flat
        else:
            fluxes[:, :self.dim] += P_n


class KTDiffusion(FVMFacetFlux):
    """ Diffusion term from K-T solver.
        Flux = a/2 * (U_L - U_R) * facet_len
     """
    E_props: FacetCalc

    def __init__(self, E_props: FacetCalc, v_factor, phy_setup: FluidConstitution, device="cpu"):
        self.device = device
        self.v_factor = v_factor
        self.E_props = E_props
        self.phy_setup = phy_setup
        self.a_clip = 1

    def facet_fluxes(self, fluxes=None):
        E_props = self.E_props
        rho_face = E_props.rho_facet
        Vs_face = E_props.Vs_facet                      # shape = [n_facets, facets=2, n_comp=dim]
        Q_face = E_props.Q_facet                        # shape = [n_facets, facets=2, n_comp=1]
        mom_face = E_props.mom_facet
        facet_size = E_props.mesh.facet_size

        # Wavespeed is c + v_max. Clip velocity wavespeed to k*c + v_max
        Vs = Vs_face.norm(dim=-1)                       # shape = [n_facets, facets=2]
        Vs_max = Vs.max(dim=1, keepdim=True).values     # shape = [n_facets, 1]
        c = self.phy_setup.c.max(dim=1).values          # shape = [n_facets, 1]

        # Reduce diffusion for velocity for low Mach number flows
        M = (Vs_max / (c + 1e-8)).abs()
        v_factor = torch.clamp(M, min=self.v_factor, max=1)

        # Use different diffusion for velocity and other components. For velocity, use v_factor * c, for others use c.
        a_vel = v_factor * c + Vs_max                   # shape = [n_facets, n_comp]
        a_other = c + Vs_max                            # shape = [n_facets, n_comp]

        # Flux = a/2 * (U_L - U_R) * facet_len
        dmom = mom_face[:, 0] - mom_face[:, 1]          # [n_facets, dim]
        drho = rho_face[:, 0] - rho_face[:, 1]          # [n_facets, 1]
        dQ = Q_face[:, 0] - Q_face[:, 1]                # [n_facets, 1]

        kt_fluxes = 0.5 * torch.cat([a_vel * dmom, a_other * drho, a_other * dQ], dim=1) * facet_size

        if fluxes is None:
            return kt_fluxes
        else:
            fluxes += kt_fluxes


class FVMEquation:
    mesh: FVMMesh
    E_props: FacetCalc
    cells: FVMCells
    n_comp: int

    def __init__(self, cfg: ConfigFVM, phy_setup: FluidConstitution, mesh: FVMMesh, n_comp, bc_tag, us_init=None):
        self.cfg = cfg
        self.phy_setup = phy_setup
        self.device = cfg.device
        self.mesh = mesh
        self.n_comp = n_comp

        device = self.device

        # Physical parameters
        E_props = FacetCalc(self.phy_setup, cfg, mesh, n_comp, bc_tag, device=device)
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
        fluxes = self.U_advect.facet_fluxes()
        # Pressure term
        self.P_force.facet_fluxes(fluxes)
        # Viscosity term
        self.U_visc.facet_fluxes(fluxes)
        # Heating term
        self.Heat.facet_fluxes(fluxes)
        # MUSCL term
        self.KT_diff.facet_fluxes(fluxes)
        # Compute divergence
        divergence = self._flux_to_div(fluxes)

        return divergence

    def _flux_to_div(self, fluxes):
        """ Compute cell divergence using fluxes.
            fluxes.shape = (n_edges, N_component)

            du/dt = -div(flux) = -sum_i (sign_i * flux_i)
        """
        divergence = self.flux_mat.spMM(fluxes)  # shape: (n_cells, n_component)
        return divergence

    def plot_flux(self, fluxes, title="Fluxes", show_index=False, lims=None, Xlims=None):
        plot_edges(self.mesh.vertices.cpu(), self.mesh.facets.cpu(), title=title, colors=fluxes, show_index=show_index, lims=lims, Xlims=Xlims)

    def plot_cells(self, values, title="Cell Values", show_index=False, lims=None, Xlims=None):
        plot_points(self.mesh.centroids.cpu(), values.T, show_index=show_index, title=title, lims=lims, Xlims=Xlims)

    def plot_interp(self, values, title="Cell Values", Xlims=None, resolution=2000):
        plot_interp_cell(self.mesh.vertices, values.T, self.mesh.cells, title=title, Xlims=Xlims)
