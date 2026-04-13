from dataclasses import dataclass
from abc import ABC
from enum import Enum


class BCMode(Enum):
    Isentropic = "Isentropic"
    Characteristic = "characteristic"
    Farfield = "Farfield"
    FarfieldBlended = "Farfield_Blended"


class ConfigBC(ABC):
    mode: BCMode
    # Farfield physical parameters
    v_n_inf: float
    v_t_inf: float
    rho_inf: float
    T_inf: float


@dataclass
class ConfigFVM(ABC):
    device: str = "cpu"
    compile: bool = True

    problem_setup: str | None = None    # {ellipse, nozzle}
    N_comp: int = 4     # Number of components in the state vector (e.g., [momentum_x, momentum_y, density, energy])

    # Temporal solver parameters
    solver_name: str = "Butcher_adapt"
    solver_extra: str = "RK3_SSP4"
    dt: float | None = None
    n_iter: int | None = None     # Max number of iterations

    # mesh parameters
    min_A: float | None = None
    max_A: float | None = None
    lnscale: float | None = None

    # Save configuration
    plot: bool = True      # Set False to disable matplotlib windows
    plot_t: float | None = None   # Time interval between plots
    save_t: float | None = None    # Time interval between saves
    save_dir: str | None = None    # Override output directory (default: auto timestamp under artefacts/fvm_saves)
    exact_interval: bool = False  # If True, clamp dt to land exactly on save_t boundaries
    print_i: int | None = None   # Iterations between print statements
    end_t: float | None = None       # Max simulation time.

    # Physical parameters, to be overwritten
    T_0: float | None = None        # Reference temperature
    viscosity: float | None = None     # At reference temp
    visc_bulk: float | None = None
    thermal_cond: float | None = None
    S_const: float | None = None       # Sutherland's constant
    gamma: float | None = None  # Ratio of specific heats
    C_v: float | None = None     # Specific heat at constant volume

    # Stability parameters
    v_factor: float = 0.1     # Clamp KT diffusion term to v_factor * c to reduce viscosity.
    lim_p: int = 4          # Order of limiter (1 for BJ)
    lim_K: float = 0.1

    # Boundary Configuration
    exit_cfg: ConfigBC | None = None
    inlet_cfg: ConfigBC | None = None


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
    problem_setup = "ellipse"    # {ellipse, nozzle}

    # Temporal solver parameters
    dt= 1e-4
    n_iter = 50000     # Max number of iterations

    # mesh parameters
    min_A = 0.25e-3
    max_A = 0.5e-3
    lnscale = 2

    # Save configuration
    plot_t = 0.5   # Time interval between plots
    save_t = 0.5    # Time interval between saves
    print_i = 500   # Iterations between print statements
    end_t = 20       # Max simulation time.

    # Physical parameters
    T_0 = 100        # Reference temperature
    viscosity = 1e-3     # At reference temp
    visc_bulk = 10e-3
    thermal_cond = 1e-6
    S_const = 110.4       # Sutherland's constant
    gamma = 1.2  # Ratio of specific heats
    C_v = 2     # Specific heat at constant volume

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
    problem_setup = "nozzle"

    # Temporal solver parameters
    dt = 1e-4
    n_iter = 50000     # Max number of iterations

    # Save configuration
    plot_t = 0.1   # Time interval between plots
    save_t = 0.5    # Time interval between saves
    print_i = 500   # Iterations between print statements
    end_t = 20       # Max simulation time.

    # mesh parameters
    min_A  = 0.5e-3
    max_A = 1e-3
    lnscale = 2

    # Physical parameters
    T_0 = 100        # Reference temperature
    viscosity = 5e-3     # At reference temp
    visc_bulk = 50e-5
    thermal_cond = 1e-6
    S_const = 110.4       # Sutherland's constant
    gamma = 1.2  # Ratio of specific heats
    C_v = 2     # Specific heat at constant volume

    def __post_init__(self):
        self.exit_cfg = NozzleFarfield()
        self.inlet_cfg = NozzleInlet()
