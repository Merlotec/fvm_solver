from dataclasses import dataclass
from abc import ABC
from enum import Enum


class BCMode(Enum):
    Isentropic = "Isentropic"
    Characteristic = "characteristic"
    Farfield = "Farfield"


@dataclass
class ConfigFVM(ABC):
    device: str = "cuda"
    compile: bool = False
    profile: bool = False         # Used for profiling code.

    problem_setup: str = None    # {ellipse, nozzle}
    N_comp: int = 4     # Number of components in the state vector (e.g., [momentum_x, momentum_y, density, energy])

    # Temporal solver parameters
    solver_name: str = "Butcher_adapt"
    solver_extra: str = "RK3_SSP4"
    dt: float = None
    n_iter: int = None     # Max number of iterations

    # mesh parameters
    min_A: float = None
    max_A: float = None
    lnscale: float = None

    # Save configuration
    plot_t: float = None   # Time interval between plots
    save_t: float = None    # Time interval between saves
    print_i: int = None   # Iterations between print statements
    end_t: float = None       # Max simulation time.

    # Physical parameters, to be overwritten
    T_0: float = None        # Reference temperature
    viscosity: float = None     # At reference temp
    visc_bulk: float = None
    thermal_cond: float = None
    S_const: float = None       # Sutherland's constant
    gamma: float = None  # Ratio of specific heats
    C_v: float = None     # Specific heat at constant volume

    # Stability parameters
    v_factor: float = 0.1     # Clamp KT diffusion term to v_factor * c to reduce viscosity.
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: int = 0.1

    # Boundary Configuration
    exit_cfg: ConfigBC = None
    inlet_cfg: ConfigBC = None


class ConfigBC(ABC):
    mode: BCMode
    # Farfield physical parameters
    v_n_inf: float
    v_t_inf: float
    v_inf: tuple[float, ...]         # Use for 3D
    rho_inf: float
    T_inf: float


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
    problem_setup: str = "ellipse"    # {ellipse, nozzle}

    # Temporal solver parameters
    dt: float = 1e-4
    n_iter: int = 50000     # Max number of iterations

    # mesh parameters
    min_A: float = 1e-5
    max_A: float = 1e-5
    lnscale: float = 2

    # Save configuration
    plot_t: float = 0.025   # Time interval between plots
    save_t: float = 0.05    # Time interval between saves
    print_i: int = 500   # Iterations between print statements
    end_t: float = 20       # Max simulation time.

    # Physical parameters
    T_0: float = 100        # Reference temperature
    viscosity: float = 10e-4     # At reference temp
    visc_bulk: float = 5e-4
    thermal_cond: float = 1e-6
    S_const: float = 110.4       # Sutherland's constant
    gamma: float = 1.2  # Ratio of specific heats
    C_v: float = 2     # Specific heat at constant volume

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
    problem_setup: str = "nozzle"

    # Temporal solver parameters
    dt: float = 1e-4
    n_iter: int = 50000     # Max number of iterations

    # Save configuration
    plot_t: float = 0.1   # Time interval between plots
    save_t: float = 0.5    # Time interval between saves
    print_i: int = 500   # Iterations between print statements
    end_t: float = 20       # Max simulation time.

    # mesh parameters
    min_A: float = 0.5e-3
    max_A: float = 1e-3
    lnscale: float = 2

    # Physical parameters
    T_0: float = 100        # Reference temperature
    viscosity: float = 5e-3     # At reference temp
    visc_bulk: float = 50e-5
    thermal_cond: float = 1e-6
    S_const: float = 110.4       # Sutherland's constant
    gamma: float = 1.2  # Ratio of specific heats
    C_v: float = 2     # Specific heat at constant volume

    def __post_init__(self):
        self.exit_cfg = NozzleFarfield()
        self.inlet_cfg = NozzleInlet()
