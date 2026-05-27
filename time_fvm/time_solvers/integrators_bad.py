""" Worse integrators. """
from __future__ import annotations
from collections import deque
from typing import TYPE_CHECKING

from time_fvm.config_fvm import ConfigFVM
from time_fvm.time_solvers.t_solvers import TSolver, FVMCells
from time_fvm.time_solvers.integrators import Adaptive
if TYPE_CHECKING:
    from time_fvm import FVMEquation


class RK3_SSP(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_b = 3/4 * U_i + 1/4 [U_a + dt * f(U_a)]
            U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        """

        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_b = 3/4 * U_i + 1/4 * dt * f(U_a)
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_b = 3/4 * U_i + 1/4 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))

        # U_{i+1} = 1/3 * U_i + 2/3 [U_b + dt * f(U_b)]
        prim_b, U_b = self.cells.convert_state_to_value(U_b)
        U_i_1 = 1/3 * U_i + 2/3 * (U_b + self.dt * self.eq.forward(prim_b, t+self.dt/2))
        return U_i_1


class RK2_SSP(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = U_i + dt * f(U_i, t)
            U_{i+1} = 0.5 * U_i + 0.5 [U_a + dt * f(U_a)]
        """
        prim_i, U_i = self.cells.get_values()
        # U_a = U_i + dt * f(U_i)
        U_a = U_i + self.dt * self.eq.forward(prim_i, t)

        # U_{i+1} = 0.5 * U_i + 0.5 * [U_a + dt * f(U_a)]
        prim_a, U_a = self.cells.convert_state_to_value(U_a)
        U_i_1 = 0.5 * U_i + 0.5 * (U_a + self.dt * self.eq.forward(prim_a, t+self.dt))
        return U_i_1


class RK2_SSP3(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
            U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
            U_{i+1} = 1/3 * U_i + 1/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        """
        _, U_i = self.cells.get_values()
        # U_a = 1/2 * U_i + 1/2 * [U_i + dt * f(U_i)]
        U_a = 1/2 * U_i + 1/2 * self._euler_step(U_i, t=t)

        # U_b = 1/2 * U_a + 1/2 * [U_a + dt * f(U_a)]
        U_b = 1/2 * U_a + 1/2 * self._euler_step(U_a, t=t+self.dt/2)

        # U_{i+1} = 1/3 * U_i + 1/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        U_i_1 = 1/3 * U_i + 1/3 * U_b + 1/3 * self._euler_step(U_b, t=t+self.dt)

        return U_i_1


class RK2_SSP4(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_a = 2/3 * U_i + 1/3 * [U_i + dt * f(U_i)]
            U_b = 2/3 * U_a + 1/3 * [U_a + dt * f(U_a)]
            U_c = 2/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
            U_{i+1} = 1/4 * U_i + 1/2 * U_c + 1/4 * [U_c + dt * f(U_c)]
        """
        _, U_i = self.cells.get_values()
        # U_a = 2/3 * U_i + 1/3 * [U_i + dt * f(U_i)]
        U_a = 2/3 * U_i + 1/3 * self._euler_step(U_i, t=t)

        # U_b = 2/3 * U_a + 1/3 * [U_a + dt * f(U_a)]
        U_b = 2/3 * U_a + 1/3 * self._euler_step(U_a, t=t+self.dt/3)

        # U_c = 2/3 * U_b + 1/3 * [U_b + dt * f(U_b)]
        U_c = 2 / 3 * U_b + 1/3 * self._euler_step(U_b, t=t + self.dt*2/3)

        # U_{i+1} = 1/4 * U_i + 1/2 * U_c + 1/4 * [U_c + dt * f(U_c)]
        U_i_1 = 1/4 * U_i + 1/2 * U_c + 1/4 * self._euler_step(U_c, t=t+self.dt)

        return U_i_1


class Leapfrog2(TSolver):
    """ Leapfrog 2 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.U_tm1 = None

    def _step(self, t):
        """
        U_{t+1} = U_t-1 + dt * f(U_t, t)
        """
        prim_t, U_t = self.cells.get_values()
        dUdt_t = self.eq.forward(prim_t, t)

        # First iteration, use Euler step
        if self.U_tm1 is None:
            self.U_tm1 = U_t + self.dt * dUdt_t
            return self.U_tm1

        # U_{t+1} = U_t-1 + dt * f(U_t, t)
        U_t_1 = self.U_tm1 + 2 *self.dt * dUdt_t

        self.U_tm1 = U_t

        return U_t_1


class LeapfrogAss(TSolver):
    """ Asselin leapfrog 1 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.U_hat_tm1 = None

        self.even=True
    def _step(self, t):
        """
        U_{t+1} = U_t-1 + dt * f(U_t, t)
        """
        prim_t, U_t = self.cells.get_values()
        dUdt_t = self.eq.forward(prim_t, t)

        # First iteration, use Euler step
        if self.U_hat_tm1 is None:
            self.U_hat_tm1 =  U_t
            U_t_1 = U_t + self.dt * dUdt_t
            return U_t_1

        # U_{t+1} = U_t-1 + dt * f(U_t, t)
        U_t_1 = self.U_hat_tm1 + 2* self.dt * dUdt_t

        self.U_hat_tm1 = U_t + 0.6 * (self.U_hat_tm1 - 2 * U_t + U_t_1)

        return U_t_1


class Magazenkov(TSolver):
    """ Leapfrog + Adams-Bashforth 2 solver
        Non-Markov solver.
        Note: Takes two half-steps
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation
        self.dt = self.dt / 2 # Half step

        # self.prev_dUdt = deque(maxlen=2)

        self.U_tm1 = None
    #
    # def _init_states(self, t):
    #     prim, _ = self.cells.get_values()
    #     dUdt_0 = self.eq.forward(prim, t)
    #
    #     for _ in range(1):
    #         self.prev_dUdt.append(dUdt_0)

    def _step(self, t):
        """
        U_{t+1} = U_{t-1} + 2 * dt * f(U_t, t)
        U_{t+2} = U_t + dt/2 * [3 * f(U_{t+1}) - f(U_{t})]
        """
        prim_t, U_t = self.cells.get_values()

        # First iteration, use Euler step
        if self.U_tm1 is None:
            self.U_tm1 = U_t # + self.dt * dUdt_t

        dUdt_t = self.eq.forward(prim_t, t)
        # U_{t+1} = U_{t-1} + 2 * dt * f(U_t, t)
        U_t_1 = self.U_tm1 + 2 * self.dt * dUdt_t

        dUdt_t1 = self.eq.forward(prim_t, t + self.dt)
        # U_{t+2} = U_t + dt/2 * [3 * f(U_{t+1}) - f(U_{t})]
        U_t_2 = U_t_1 + self.dt/ 2 * (3 * dUdt_t1 - dUdt_t)

        self.U_tm1 = U_t_1
        return U_t_2


class Adams2(TSolver, Adaptive):
    """ Adams Bashforth 2 solver
        Non-Markov solver.
    """
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

        self.prev_dUdt = deque(maxlen=2)
        self._adapt_init(order=4, atol=3e-6, rtol=3e-6, mtol=1e-7, alphas=(0.9, 0.99), device=cfg.device)

    def _init_states(self, t):
        prim, _ = self.cells.get_values()
        dUdt_0 = self.eq.forward(prim, t)

        for _ in range(2):
            self.prev_dUdt.append(dUdt_0)

    def _step(self, t):
        """
        U_{t+1} = U_t + dt/2 * [3 * f(U_t) - f(U_{t-1})]
        :param t:
        :return:
        """
        if len(self.prev_dUdt) == 0:
            self._init_states(t)

        prim_t, U_t = self.cells.get_values()

        dUdt_t = self.eq.forward(prim_t, t)
        dUdt_tm1 = self.prev_dUdt[-1]
        dUdt_tm2 = self.prev_dUdt[-2]

        U_1_high = U_t + self.dt/12 * (23 * dUdt_t - 16 * dUdt_tm1 + 5 * dUdt_tm2)
        U_1_low = U_t + self.dt/2 * (3 * dUdt_t - dUdt_tm1)
        # U_1_low = U_t + self.dt * dUdt_t
        self.prev_dUdt.append(dUdt_t)
        self.update_stepsize(U_1_high, U_1_low, U_t)

        return U_1_high


class ExplMidpoint(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U^{i+0.5} = U^i + dt/2 * f(U^i)
            U^{i+1} = U^i + dt * f(U^{i+0.5})
        """

        state = self.cells.state
        primatives, _ = self.cells.get_values()

        dUdt_star = self.eq.forward(primatives, t=t)
        U_star = state + 0.5 * self.dt * dUdt_star        # U_{i+0.5}

        primatives_star, _ = self.cells.convert_state_to_value(U_star)
        dUdt = self.eq.forward(primatives_star, t=t)
        U_i_1 = state + self.dt * dUdt

        return U_i_1


class Heuns(TSolver):
    def __init__(self, cells: FVMCells, equation, cfg: ConfigFVM):
        super().__init__(cells, eq=equation, cfg=cfg)
        self.eq: FVMEquation = equation

    def _step(self, t):
        """ U_s = U_i + dt * f(U_i)
            U_{i+1} = U_i + dt * [0.5 * f(U_s}) + 0.5 * f(U_i)]
        """
        U_0 = self.cells.state
        # y_star = y_n + dt*f(t_n, y_n)
        dUdt_star = self._forward_state(U_0)
        U_star = U_0 + self.dt * dUdt_star

        # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        dUdt = self._forward_state(U_star)
        U_i_1 = self.cells.state + 0.5 * self.dt * (dUdt_star + dUdt)

        # # y_{n+1} = y_n + 0.5*dt*[f(t_n, y_n) + f(t_n+1, y_star)]
        # dUdt = self._forward_state(U_i_1, t)
        # U_i_1 = self.cells.state + 0.5 * self.dt * (dUdt_star + dUdt)

        return U_i_1



