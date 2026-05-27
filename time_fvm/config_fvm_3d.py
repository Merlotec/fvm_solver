from dataclasses import dataclass

from time_fvm.config_fvm import ConfigFVM, BCMode, ConfigBC



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
    v_inf = (0, 0, 0)
    rho_inf = 1
    T_inf = 100


@dataclass
class ConfigEllipse(ConfigFVM):
    N_comp: int = 5     # Number of components in the state vector (e.g., [momentum_x, momentum_y, density, energy])

    problem_setup: str = "ellipse"    # {ellipse, nozzle}

    # Temporal solver parameters
    dt: float = 1e-4
    n_iter: int = 50000     # Max number of iterations

    # mesh parameters
    min_A: float = 0.05e-3
    max_A: float = 0.05e-3
    lnscale: float = 2

    # Save configuration
    plot_t: float = 0.025   # Time interval between plots
    save_t: float = 0.05    # Time interval between saves
    print_i: int = 500   # Iterations between print statements
    end_t: float = 20       # Max simulation time.

    # Physical parameters
    T_0: float = 100        # Reference temperature
    viscosity: float = 1e-6     # At reference temp
    visc_bulk: float = 1e-4
    thermal_cond: float = 1e-6
    S_const: float = 110.4       # Sutherland's constant
    gamma: float = 1.2  # Ratio of specific heats
    C_v: float = 2     # Specific heat at constant volume

    def __post_init__(self):
        self.exit_cfg = EllipseFarfield()
        self.inlet_cfg = EllipseInlet()


