from typing import Optional, Union
from dataclasses import dataclass
from abc import ABC
from enum import Enum


class BCMode(Enum):
    Isentropic = "Isentropic"
    Characteristic = "characteristic"
    Farfield = "Farfield"


class ViscosityModel(Enum):
    Newtonian = "Newton"                    # constant viscosity
    PowerLaw = "PowerLaw"                    # ~ gamma_dot^(n-1)
    Carreau = "Carreau"                     # interpolate between two viscosities
    HerschelBulkley = "HerschelBulkley"     # step between two viscosities


class ConfigBC(ABC):
    mode: BCMode
    # Farfield physical parameters
    v_n_inf: float
    v_t_inf: float
    v_inf: tuple[float, ...]         # Use for 3D
    rho_inf: float
    T_inf: float


@dataclass
class ConfigFVM(ABC):
    device: str = "cpu"
    compile: bool = True
    profile: bool = False

    problem_setup: Optional[str] = None    # {ellipse, nozzle}
    N_comp: int = 4     # Number of components in the state vector (e.g., [momentum_x, momentum_y, density, energy])

    # Temporal solver parameters
    solver_name: str = "Butcher_adapt"
    solver_extra: str = "RK3_SSP4"
    dt: Optional[float] = None
    n_iter: Optional[int] = None     # Max number of iterations

    # mesh parameters
    min_A: Optional[float] = None
    max_A: Optional[float] = None
    lnscale: Optional[float] = None

    # Save configuration
    plot: bool = True      # Set False to disable matplotlib windows
    plot_t: Optional[float] = None   # Time interval between plots
    save_t: Optional[float] = None    # Time interval between saves
    save_dir: Optional[str] = None    # Override output directory (default: auto timestamp under artefacts/fvm_saves)
    exact_interval: bool = False  # If True, clamp dt to land exactly on save_t boundaries
    print_i: Optional[int] = None   # Iterations between print statements
    end_t: Optional[float] = None       # Max simulation time.

    # Physical parameters, to be overwritten
    T_0: Optional[float] = None        # Reference temperature
    viscosity: Optional[float] = None     # At reference temp
    visc_bulk: Optional[float] = None
    thermal_cond: Optional[float] = None
    S_const: Optional[float] = None       # Sutherland's constant
    gamma: Optional[float] = None  # Ratio of specific heats
    C_v: Optional[float] = None     # Specific heat at constant volume

    # Non-Newtonian viscosity (see FluidConstitution2D._viscosity_coef).  Defaults
    # give Newtonian behaviour; the generator sweeps these for shear-thinning /
    # thickening / yield-stress fluids.  visc_n/min_factor/gamma_scale are only read
    # for the non-Newtonian models.
    visc_model: ViscosityModel = ViscosityModel.Newtonian
    visc_n: Optional[float] = 1.0             # power-law index (1 Newtonian, <1 thinning, >1 thickening)
    visc_min_factor: Optional[float] = 0.1    # low/high-shear viscosity ratio bound
    visc_gamma_scale: Optional[float] = 10.0  # shear-rate scale where non-Newtonian effects kick in

    # Stability parameters
    v_factor: float = 0.1     # Clamp KT diffusion term to v_factor * c to reduce viscosity.
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: float = 0.1

    # Boundary Configuration
    exit_cfg: Optional[ConfigBC] = None
    inlet_cfg: Optional[ConfigBC] = None


# ------------------------------- Ellipse-specific configurations -------------------------------
@dataclass
class EllipseFarfield(ConfigBC):
    mode: BCMode = BCMode.Characteristic

    # Farfield physical parameters
    v_n_inf: float = -5.5
    v_t_inf: float = 0
    rho_inf: float = 1
    T_inf: float = 100


@dataclass
class EllipseInlet(ConfigBC):
    mode: BCMode = BCMode.Characteristic

    # Target inlet physical parameters
    v_n_inf = 5.5
    v_t_inf: float = 0
    rho_inf = 1
    T_inf = 100


@dataclass
class ConfigEllipse(ConfigFVM):
    problem_setup: Optional[str] = "ellipse"

    # Temporal solver parameters
    dt: Optional[float]        = 1e-4
    n_iter: Optional[int]      = 50000

    # mesh parameters
    min_A: Optional[float]     = 0.25e-3
    max_A: Optional[float]     = 0.5e-3
    lnscale: Optional[float]   = 2

    # Save configuration
    plot_t: Optional[float]    = 0.5
    save_t: Optional[float]    = 0.5
    print_i: Optional[int]     = 500
    end_t: Optional[float]     = 20

    # Physical parameters
    T_0: Optional[float]           = 100
    viscosity: Optional[float]     = 1e-3
    visc_bulk: Optional[float]     = 10e-3
    thermal_cond: Optional[float]  = 1e-6
    S_const: Optional[float]       = 110.4
    gamma: Optional[float]         = 1.2
    C_v: Optional[float]           = 2

    def __post_init__(self):
        self.exit_cfg = EllipseFarfield()
        self.inlet_cfg = EllipseInlet()

# ------------------------------- Nozzle-specific configurations -------------------------------
@dataclass
class NozzleFarfield(ConfigBC):
    mode: BCMode = BCMode.Characteristic

    # Farfield physical parameters
    v_n_inf: float = 0
    v_t_inf: float = 0
    rho_inf: float = 1
    T_inf: float = 100


@dataclass
class NozzleInlet(ConfigBC):
    mode: BCMode = BCMode.Characteristic

    # Target inlet physical parameters
    v_n_inf: float = 0
    v_t_inf: float = 0
    rho_inf: float = 2.5
    T_inf: float = 400


@dataclass
class ConfigNozzle(ConfigFVM):
    problem_setup: Optional[str] = "nozzle"

    # Temporal solver parameters
    dt: Optional[float]        = 1e-4
    n_iter: Optional[int]      = 50000

    # mesh parameters
    min_A: Optional[float]     = 0.5e-3
    max_A: Optional[float]     = 1e-3
    lnscale: Optional[float]   = 2

    # Save configuration
    plot_t: Optional[float]    = 0.1
    save_t: Optional[float]    = 0.5
    print_i: Optional[int]     = 500
    end_t: Optional[float]     = 20

    # Physical parameters
    T_0: Optional[float]           = 100
    viscosity: Optional[float]     = 5e-3
    visc_bulk: Optional[float]     = 50e-5
    thermal_cond: Optional[float]  = 1e-6
    S_const: Optional[float]       = 110.4
    gamma: Optional[float]         = 1.2
    C_v: Optional[float]           = 2

    def __post_init__(self):
        self.exit_cfg = NozzleFarfield()
        self.inlet_cfg = NozzleInlet()
