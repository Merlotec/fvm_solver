from __future__ import annotations
from typing import TYPE_CHECKING
from collections import deque
import torch

from time_fvm.t_solvers import TSolver, FVMCells
if TYPE_CHECKING:
    from time_fvm.fvm_equation import FVMEquation
    from time_fvm.config_fvm import ConfigFVM

def get_solver(cells: FVMCells, equation: FVMEquation, cfg: ConfigFVM) -> TSolver:
    name = cfg.solver_name
    extra_str = cfg.solver_extra
    if name == "RK3_SSP4":
        return RK3_SSP4(cells, equation, cfg)
    elif name == "Adams3_PC":
        return Adams3PC(cells, equation, cfg)
    elif name == "Adams4_PC":
        return Adams4PC(cells, equation, cfg)
    elif name == "Butcher_adapt":
        return ButcherAdapt(cells, equation, extra_str, cfg)
    elif name == "Butcher":
        return Butcher(cells, equation, extra_str, cfg)
    elif name == "Euler":
        return Euler(cells, equation, cfg)
    else:
        raise NotImplementedError("Invalid solver.")

class Adaptive:
    def _adapt_init(self, order: int, rtol, atol, mtol, alphas, dt_min=None, dt_max=None, device="cuda"):
        self.order = order
        self.rtol = rtol
        self.atol = atol
        self.mtol = mtol
        self.alphas = torch.tensor(alphas, device=device)

        self.dt_min = dt_min
        self.dt_max = dt_max

    def update_stepsize(self, dU_high, dU_low, Us):
        """ Update the time step size based on the difference between two solutions.
            U.shape = [n_cells, n_comp]

            dU_high: High accuracy solver prediction
            dU_low: Low accuracy solver prediction
            Us: Sate. Used for scaling the error.
        """
        # Compute the difference between the two solutions
        diff = dU_high - dU_low

        # Compute a new time step size based on the difference
        # For example, you could use a simple heuristic like:
        E = torch.norm(diff, dim=0) / (self.atol + self.rtol * torch.norm(dU_high, dim=0) + self.mtol * torch.norm(Us, dim=0))
        E = E.mean()

        # If E<1, increase the time step size, otherwise decrease step size
        factor = 0.9 * (1 / E) ** (1 / self.order)
        alpha = torch.where(E > 1, self.alphas[0], self.alphas[1])

        self.dt = self.dt * (alpha  + (1-alpha) * factor)


class Butcher_Tables:
    def __init__(self, name, device):
        if name == "RK4":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=torch.float32)

            b = torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=torch.float32)
            c = torch.tensor([0.0, 0.5, 0.5, 1.0], dtype=torch.float32)

        elif name == "RK3_SSP4":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [1 / 6, 1 / 6, 1 / 6, 0.0]
            ], dtype=torch.float32)

            b = torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 2], dtype=torch.float32)
            c = torch.tensor([0.0, 0.5, 1, 0.5], dtype=torch.float32)
            b2 = torch.tensor([1 / 4, 1 / 4, 1 / 4, 1 / 4], dtype=torch.float32)
            self.b2 = b2.reshape(-1, 1, 1)

        elif name == "RK3_SSP5":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0],
                [0.37726891511710, 0.0, 0.0, 0.0, 0],
                [0.37726891511710, 0.37726891511710, 0.0, 0.0, 0],
                [0.16352294089771, 0.16352294089771, 0.16352294089771, 0.0, 0],
                [0.14904059394856, 0.14831273384724, 0.14831273384724, 0.34217696850008, 0],
            ], dtype=torch.float32)

            b = torch.tensor([0.19707596384481, 0.11780316509765, 0.11709725193772, 0.27015874934251, 0.29786487010104], dtype=torch.float32)
            c = torch.tensor([0, 0.37726891511710 , 0.75453783023419 , 0.49056882269314 , 0.78784303014311 ], dtype=torch.float32)
            b2 = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5], dtype=torch.float32)

        elif name == """RK3_SSP6""":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0, 0],
                [0.28422, 0.0, 0.0, 0.0, 0, 0],
                [0.28422, 0.28422, 0.0, 0.0, 0, 0],
                [0.23071, 0.23071, 0.23071, 0.0, 0, 0],
                [0.13416, 0.13416, 0.13416, 0.16528, 0, 0],
                [0.13416, 0.13416, 0.13416, 0.16528, 0.28422, 0]
            ], dtype=torch.float32)

            b = torch.tensor([0.17016,  0.17016,  0.10198,  0.12563,  0.21604,  0.21604], dtype=torch.float32)
            c = torch.tensor([0, 0.28422, 0.56844 , 0.69213, 0.56776, 0.85198], dtype=torch.float32)
            b2 = torch.tensor([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], dtype=torch.float32)

        elif name == """RK4_SSP5""":
            A = torch.tensor([
                [0.0, 0.0, 0.0, 0.0, 0],
                [0.39175222700392, 0.0, 0.0, 0.0, 0],
                [0.21766909633821, 0.36841059262959, 0.0, 0.0, 0],
                [0.08269208670950, 0.13995850206999,  0.25189177424738, 0.0, 0],
                [0.06796628370320, 0.11503469844438, 0.20703489864929, 0.54497475021237, 0],
            ], dtype=torch.float32)

            b = torch.tensor([0.14681187618661, 0.24848290924556, 0.10425883036650, 0.27443890091960, 0.22600748319395], dtype=torch.float32)
            c = torch.tensor([0., 0.39175222700392, 0.58607968896779 , 0.47454236302687, 0.93501063100924], dtype=torch.float32)

        elif name == "RK4_SSP10":
            A = torch.tensor([
                [0]*10,
                [1/6] + [0]*9,
                [1/6]*2 + [0]*8,
                [1/6]*3 + [0]*7,
                [1/6]*4 + [0]*6,
                [1/15]*5 + [0]*5,
                [1 / 15] * 5 + [1/6] + [0]*4,
                [1 / 15] * 5 + [1/6]*2 + [0]*3,
                [1 / 15] * 5 + [1/6]*3 + [0]*2,
                [1 / 15] * 5 + [1/6]*4 + [0]*1,

            ], dtype=torch.float32)
            b = torch.tensor([1/10]*10, dtype=torch.float32)
            c = A.sum(dim=1)

            b2 = torch.tensor([1/5, 0, 0, 3/10, 0, 0, 1/5, 0, 3/10, 0], dtype=torch.float32)
        else:
            raise NotImplementedError("Unknown Butcher tableau")

        self.A, self.b, self.c = A.to(device), b.reshape(-1, 1, 1).to(device), c.to(device)
        self.b2 = b2.reshape(-1, 1, 1).to(device) if 'b2' in locals() else None


class RK3_SSP4(TSolver, Adaptive):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self._adapt_init(order=4, atol=2e-3, rtol=2e-3, mtol=1e-6, alphas=(0.8, 0.995), dt_min=self.dt*0.5, device=cfg.device)

    def _step(self, t):
        """ U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
            U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
            U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
            U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        """

        U_0 = self.cells.state
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * (U_0 + self._euler_step(U_0, t=t))

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * (U_a + self._euler_step(U_a, t=t+self.dt/2))

        # U_c = 2/3 * U_i + 1/6 * U_b + 1/6 * [U_b + dt * f(U_b)]
        U_c = 2/3 * U_0 + 1/6 * (U_b + self._euler_step(U_b, t=t+self.dt))

        # U_{i+1} = 1/2 * U_c + 1/2 [U_c + dt * f(U_c)]
        U_i_1 = 1/2 * (U_c + self._euler_step(U_c, t=t+self.dt/2))

        self.update_stepsize((U_i_1 - U_0), (U_b - U_0), U_0)

        return U_i_1


class Adams3PC(TSolver, Adaptive):
    """ Adams–Bashforth–Moulton predictor corrector 3 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=2e-3, rtol=2e-3, mtol=1e-7, alphas=(0.9, 0.99), dt_min=self.dt/2, device=cfg.device)
        self.need_init = True

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, self.dt, t)

        self.dUdt_m1 = dUdt_0
        self.dUdt_m2 = dUdt_0


    def _step(self, t):
        """
        U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        :param t:
        :return:
        """
        if self.need_init:
            self._init_states(t)
            self.need_init = False

        prim_t, U_0 = self.cells.get_values()

        dUdt_0 = self.eq.forward(prim_t, self.dt, t)
        dUdt_m1 = self.dUdt_m1
        dUdt_m2 = self.dUdt_m2

        # U_a = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        # U_ac = U_0 + self.dt * dUdt_0
        # U_a = U_0 + self.dt/2 * (3 * dUdt_0 - dUdt_m1)
        # U_a = U_0 + self.dt / 12 * (23 * dUdt_0 - 16 * dUdt_m1 + 6 * dUdt_m2)
        # U_a = 1/3 * U_a  + 1/3 * U_ab  + 1/3 * U_ac
        U_a = U_0 + self.dt / 36 * (53 * dUdt_0 - 22 * dUdt_m1 + 6 * dUdt_m2)

        # U_{t+1} = U_t + dt/12 * [5 * f(U_a) + 8 * f(U_{t}) - 1 * f(U_{t-1})]
        dUdt_a = self._forward_state(U_a, t)
        dU_high = self.dt/12 * (5 * dUdt_a + 8 * dUdt_0 - dUdt_m1)
        dU_low = self.dt/2 * (dUdt_a + dUdt_0)
        U_1_high =  U_0 + dU_high


        # Update buffer
        self.dUdt_m2 = dUdt_m1
        self.dUdt_m1 = dUdt_0

        self.update_stepsize(dU_high, dU_low, U_0)

        return U_1_high


class Adams4PC(TSolver, Adaptive):
    """ Adams–Bashforth–Moulton predictor corrector 4 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=1e-3, rtol=1e-3, mtol=1e-6, alphas=(0.9, 0.99), dt_min=self.dt/2, device=cfg.device)
        self.need_init = True

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, self.dt, t)

        self.dUdt_m1 = dUdt_0
        self.dUdt_m2 = dUdt_0

    def _step(self, t):
        """
        U_a = U_t + dt/24 * [55 * f(U_t) - 59 * f(U_{t-1}) + 37 * f(U_{t-2}) - 9 * f(U_{t-3})]  (Or other order predictor)
        U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        """
        if self.need_init:
            self._init_states(t)
            self.need_init = False

        prim_t, U_0 = self.cells.get_values()

        dUdt_0 = self.eq.forward(prim_t, self.dt, t)
        dUdt_m1 = self.dUdt_m1
        dUdt_m2 = self.dUdt_m2

        # Predictor step
        # U_ac = U_t + self.dt  * dUdt_t
        #U_a = U_0 + self.dt / 2 * (3 * dUdt_0 - dUdt_m1)
        # U_a = U_0 + self.dt/12 * (23 * dUdt_0 - 16 * dUdt_m1 + 6 * dUdt_m2)
        # U_a = 1/3 * U_a  + 1/3 * U_ab  + 1/3 * U_ac
        U_a = U_0 + self.dt / 36 * (53 * dUdt_0 - 22 * dUdt_m1 + 6 * dUdt_m2)

        # U_{t+1} = U_t + dt/24 * [9 * f(U_a) + 19 * f(U_{t}) - 5 * f(U_{t-1}) + f(U_{t-2})]
        dUdt_a = self._forward_state(U_a, t)

        dU_high = self.dt / 24 * (9 * dUdt_a + 19 * dUdt_0 - 5 * dUdt_m1 + dUdt_m2)
        dU_low = self.dt / 12 * (5 * dUdt_a + 8 * dUdt_0 - dUdt_m1)
        U_1_high =  U_0 + dU_high

        # Update buffer
        self.dUdt_m2 = dUdt_m1
        self.dUdt_m1 = dUdt_0

        self.update_stepsize(dU_high, dU_low, U_0)
        return U_1_high


class ButcherAdapt(TSolver, Adaptive):
    def __init__(self, cells: FVMCells, equation, name, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        """
        Initializes the solver with a Butcher tableau.
        """

        tables = Butcher_Tables(name, cells.device)

        self.A = tables.A.unsqueeze(-1).unsqueeze(-1)
        self.b = tables.b
        self.b2 = tables.b2
        self.c = tables.b
        self.stages = self.b.shape[0]

        self._adapt_init(order=4, atol=2e-3, rtol=2e-3, mtol=1e-6, alphas=(0.8, 0.995), dt_min=self.dt*0.5, device=cfg.device)
        self.k = torch.zeros((self.stages, *self.cells.state.shape), device=self.A.device)

    def _step(self, t) -> torch.Tensor:
        """
        Take one step of the ODE solver.

        Returns:
            torch.Tensor: Updated state after one step.
        """

        state_0 = self.cells.state

          # shape = [stages, n_cells, n_comp]
        for i in range(self.stages):
            if i == 0:
                increment = 0
            else:
                # Compute the increment for y using previous stages
                increment = (self.A[i, :i] * self.k[:i]).sum(dim=0)
            # Evaluate the derivative at the stage time and state
            k_i =  self.dt * self._forward_state(state_0 + increment, t + self.c[i] * self.dt)
            self.k[i] = k_i

        # Combine stages to compute next state
        dU_high = torch.sum(self.b * self.k, dim=0)
        dU_low = torch.sum(self.b2 * self.k, dim=0)
        U_next_high = state_0 + dU_high

        self.update_stepsize(dU_high, dU_low, state_0)
        return U_next_high


class Butcher(TSolver):
    def __init__(self, cells: FVMCells, equation, name, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        """
        Initializes the solver with a Butcher tableau.

        Args:
            A (torch.Tensor): 2D tensor of stage coefficients with shape (s, s),
                              where s is the number of stages.
            b (torch.Tensor): 1D tensor of weights for combining stages.
            c (torch.Tensor): 1D tensor of time coefficients for each stage.
        """

        tables = Butcher_Tables(name, cells.device)

        self.A = tables.A
        self.b = tables.b
        self.c = tables.b
        self.stages = self.b.shape[0]

    def _step(self, t) -> torch.Tensor:
        """
        Take one step of the ODE solver.

        Returns:
            torch.Tensor: Updated state after one step.
        """

        state_0 = self.cells.state
        primatives, _ = self.cells.get_values()

        k = torch.zeros((self.stages, *state_0.shape), dtype=state_0.dtype, device=state_0.device)  # shape = [stages, n_cells, n_comp]
        for i in range(self.stages):
            if i == 0:
                increment = 0
            else:
                # Compute the increment for y using previous stages
                increment = (self.A[i, :i].unsqueeze(-1).unsqueeze(-1) * k[:i]).sum(dim=0)
            # Evaluate the derivative at the stage time and state
            k_i = self.dt * self._forward_state(state_0 + increment, t + self.c[i] * self.dt)
            k[i] = k_i

        # Combine stages to compute next state
        U_next_high = state_0 + torch.sum(self.b * k, dim=0)
        return U_next_high


class Euler(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq = equation

    def _step(self, t):
        """U^{i+1} = U^i + dt * f(U^i)"""

        dUdt = self.eq.forward(self.cells.get_values()[0], self.dt, t=t)
        U_i_1 = self.cells.state + self.dt * dUdt

        return U_i_1

