from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from time_fvm.config_fvm import ConfigFVM

class SlopeLimiter:
    def __init__(self, areas, cfg: ConfigFVM):
        lim_p = cfg.lim_p
        K = cfg.lim_K
        areas = areas.view(-1, 1, 1)

        self.eps_p = (K * areas ** 0.5) ** (lim_p+1)

        if lim_p == 1:
            self._limit = self.p1
        elif lim_p == 2:
            self._limit = self.p2
        elif lim_p == 3:
            self._limit = self.p3
        elif lim_p == 4:
            self._limit = self.p4
        elif lim_p == 5:
            self._limit = self.p5
        else:
            raise NotImplementedError(f"Limiter of order {lim_p} is not implemented.")

    def p1(self, delta, dU):
        """ BJ limiter"""
        dU = 2 * ((dU > 0).float() - 0.5) * (dU.abs() + 1e-8)
        r = delta / dU  # shape = [n_cells, neigh=3, n_comp]
        phi = torch.clamp(r, min=0., max=1)  # shape = [n_cells, neigh=3, n_comp]
        return phi

    def p2(self, delta, dU):
        """ Venkatakrishnan limiter"""
        phi = (delta ** 2 + self.eps_p + 2 * delta * dU) / (delta ** 2 + 2 * dU ** 2 + delta * dU + self.eps_p)
        return phi

    def p3(self, delta, dU):
        """ 3rd order limiter - https://arc.aiaa.org/doi/epdf/10.2514/6.2022-1374"""
        a = delta.abs()
        b = dU.abs()

        a_eps = a ** 3 + self.eps_p
        S = 4 * b ** 2
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 2 + S))
        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def p4(self, delta, dU):
        """ 4th order limiter """
        a = delta.abs()
        b = dU.abs()
        b2 = 2 * b
        a_eps = a ** 4 + self.eps_p

        # S = 2 * b * (a.square() - 2 * b * (a - 2 * b))
        # phi = (a_eps + a * S) / (a_eps + b * (delta ** 3 + S))

        S = b2 * (torch.addcmul(a.square(), b2, b2 - a))
        phi = torch.addcmul(a_eps, a, S) / torch.addcmul(a_eps, b, (delta ** 3 + S))

        phi = torch.where(a < b2, phi, 1)
        return phi

    def p5(self, delta, dU):
        """ 5th order limiter """
        a = delta.abs()
        b = dU.abs()
        a_eps = a ** 5 + self.eps_p
        S = 8 * b ** 2 * (a ** 2 - 2 * b * (a - b))
        phi = (a_eps + a * S) / (a_eps + b * (delta ** 4 + S))

        phi = torch.where(a < 2 * b, phi, 1)
        return phi

    def limit(self, delta, dU):
        """ delta: maximum allowed values
            dU: Predicted value from lstsq gradient
        """
        phi = self._limit(delta, dU)    # shape = [n_cells, neigh=3, n_comp]
        # Cell wide clamping
        phi = torch.amin(phi, dim=1, keepdim=True)  # shape = [n_cells, neigh=1, n_comp]

        return phi