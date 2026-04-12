from __future__ import annotations
from cprint import c_print
import torch
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import time

from typing import TYPE_CHECKING
from time_fvm.ds_generation.saving import Saver
if TYPE_CHECKING:
    from time_fvm.fvm_equation import FVMEquation, PhysicalSetup
    from time_fvm.config_fvm import ConfigFVM



class FVMCells:
    state: torch.Tensor  # shape = (n_cells, N_component)
    """ State stored as: [momentum_x, momenum_y, density, energy] """
    def __init__(self, n_cells, n_component, phys_setup: PhysicalSetup, init_val=None, device="cpu"):
        self.device = device
        self.phys_setup = phys_setup

        if init_val is None:
            self.state = torch.zeros(n_cells, n_component, device=device)
        else:
            assert init_val.shape == (n_cells, n_component), f'Incorrect us init shape {init_val.shape = }'
            self.state = init_val.to(device)


    def update_cells(self, state_new):
        """ Update cell values """
        self.state =  state_new

    def get_values(self):
        return self.phys_setup.state_to_primative(self.state)

    def convert_state_to_value(self, state):
        return self.phys_setup.state_to_primative(state)

    def save(self, name="state.pt"):
        torch.save(self.state, name)

    def load(self, name="state.pt"):
        self.state = torch.load(name, weights_only=True)


class TSolver(ABC):
    """
    Time-stepping solver for PDEs. This class is abstract and should be subclassed
    to implement specific time-stepping schemes.
    """
    cells: FVMCells
    eq: FVMEquation

    def __init__(self, cells: FVMCells, eq: FVMEquation, cfg: ConfigFVM):
        """
        Initialize the time-stepping solver.
        """
        self.device = cfg.device
        self.dt = cfg.dt
        self.n_steps = cfg.n_iter
        self.cells = cells
        self.eq = eq
        self.print_i = cfg.print_i
        self.plot_t = cfg.plot_t
        self.plot = cfg.plot
        self.save_t = cfg.save_t
        self.exact_interval = cfg.exact_interval
        self.saver = Saver(self.eq.E_props, save_dir=cfg.save_dir)

        # replace self._solve_step with compiled version if needed
        if cfg.compile:
            self._solve_step = torch.compile(self._solve_step)

    def _solve(self):
        """ Main loop for the program.
                - Loops through time steps, calling the step function to update the solution.
                - Prints out progress every print_i iterations.
                - Plots the solution every plot_t seconds.
                - Saves the solution every save_t seconds.
        """
        self.dt = torch.tensor(self.dt, device=self.device)
        next_plot_t = 0 # self.plot_t
        save_count = 1
        next_save_t = self.save_t  # always derived as save_count * save_t to avoid float drift

        # Warmup: trigger torch.compile tracing before the timed loop.
        c_print("Compiling...", color="bright_magenta")
        _state_backup = self.cells.state.clone()
        self._solve_step(self.dt)
        self.cells.state = _state_backup
        c_print("Compiled.", color="bright_magenta")

        # Save initial state at t=0 before any stepping
        c_print('saving: t=0', color="bright_cyan")
        primatives_init = self.cells.get_values()[0]
        self.saver.save(torch.tensor(0.0, device=self.device), self.eq.E_props, primatives_init)

        st_time = time.time()
        t, dts = 0., []
        for i in range(self.n_steps):
            # If exact_interval, clamp dt so we land exactly on the next save boundary.
            # After the clamped step, restore dt so the adaptive solver continues normally.
            if self.exact_interval and t + self.dt > next_save_t:
                dt_saved = self.dt.clone()
                self.dt = torch.tensor(next_save_t - t, device=self.device, dtype=self.dt.dtype)
                t += self.dt
                self._solve_step(t)
                self.dt = dt_saved
            else:
                t += self.dt
                self._solve_step(t)
            dts.append(self.dt)

            if i % self.print_i == 0:
                irl_time = (time.time() - st_time)/self.print_i
                avg_dt = sum(dts[-self.print_i:]) / len(dts[-self.print_i:])
                avg_dt = avg_dt.item()
                c_print(f'progress: {i = }, {t = :.4g}, {avg_dt = :.3g}, {irl_time = :.3g}', color="bright_green")
                st_time = time.time()

            if t >= next_plot_t:
                next_plot_t = t + self.plot_t
                primatives = self.cells.get_values()[0]
                if self.plot:
                    c_print(f'plot: {t = :.5g}', color="bright_yellow")
                    Xlims = None # [[3.1, 3.4], [-1.8, -1.6]] #, [(0, 1), [0, 0.5]]  #
                    titles = ["Vx", "Vy", "rho", "T"]
                    titles = [f'{title} at {t=:4g}' for title in titles]
                    self.eq.plot_interp(primatives[:, :], title=titles, Xlims=Xlims)

                if torch.any(torch.isnan(primatives)):
                    print("Nan in primatives")
                    exit(9)

            if t >= next_save_t:
                save_count += 1
                next_save_t = save_count * self.save_t
                c_print(f'saving: t={t:.5g}', color="bright_cyan")
                primatives = self.cells.get_values()[0]
                self.saver.save(t, self.eq.E_props, primatives)

        dts = torch.stack(dts).cpu()
        print(f'{dts[500:].mean() = }')
        if self.plot:
            kernel_size = 10
            kernel = torch.ones(1, 1, kernel_size) / kernel_size
            dts_smooth = torch.nn.functional.conv1d(dts.unsqueeze(0).unsqueeze(0), kernel, padding="valid")[0][0]
            plt.plot(dts)
            plt.plot(dts_smooth)
            plt.show()


    # def _save_meshio(self, primatives):
    #     """ Save as .vtu file for paraview """
    #     import meshio
    #     import glob, os
    #
    #     # Get save name
    #     os.makedirs('saves', exist_ok=True)
    #     vtu_files = glob.glob('./saves/*.vtu')
    #     number = [int(file.replace('.vtu', '').split('_')[-1]) for file in vtu_files]
    #
    #     if len(number) == 0:
    #         number = 0
    #     else:
    #         number = max(number) + 1
    #     save_name = f'./saves/save_{number:05d}.vtu'
    #     print(f'{save_name = }')
    #
    #     # Save mesh
    #     E_props = self.eq.E_props
    #     points = E_props.mesh.vertices.numpy()
    #     cells = [("triangle", E_props.mesh.triangles.numpy())]
    #     Vx, Vy, rho, T = primatives[:, 0], primatives[:, 1], primatives[:, 2], primatives[:, 3]
    #     Vx, Vy, rho, T = Vx.cpu().unsqueeze(0).numpy(), Vy.cpu().unsqueeze(0).numpy(), rho.cpu().unsqueeze(0).numpy(), T.cpu().unsqueeze(0).numpy()
    #
    #     state = self.cells.state
    #     momx, momy, E = state[:, 0], state[:, 1], state[:, 3]
    #     momx, momy, E = momx.cpu().unsqueeze(0).numpy(), momy.cpu().unsqueeze(0).numpy(), E.cpu().unsqueeze(0).numpy()
    #     P = self.eq.phy_setup.R * rho * T
    #     mesh = meshio.Mesh(
    #         points,
    #         cells,
    #         # Optionally provide extra data on points, cells, etc.
    #         cell_data={"Vx": Vx, "Vy": Vy, "P": P, "rho": rho, "T": T, "MomX": momx, "MomY": momy, "E": E, },
    #     )
    #     mesh.write(
    #         save_name,  # str, os.PathLike, or buffer/open file
    #     )

    @torch.inference_mode()
    def solve(self):
        run = True
        if run:
            self._solve()
        else:
            self._solve_profile()

    def _solve_step(self, t):
        new_Us = self._step(t)
        self.cells.update_cells(new_Us)

    def _solve_profile(self):
        import torch.profiler

        for i in range(5):
            t = i * self.dt
            self._solve_step(t)

        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()
        # tot_el = 0
        # for name, value in vars(self.eq.t_solver).items():
        #
        #     if torch.is_tensor(value) and value.is_cuda:
        #         if value.is_sparse or value.is_sparse_csr:
        #             numel = value._nnz()
        #         else:
        #             numel = value.numel()
        #         print(f"Name: {name}, Size: {value.size()}, numel = {numel}")
        #         tot_el += numel
        #
        # c_print(f'{tot_el = }', color="magenta")

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                # schedule=torch.profiler.schedule(
                #     warmup=1,  # Skip the first iteration (warm-up)
                #     wait=1,  # Skip the first iteration (warm-up)
                #     active=3  # Capture the next 3 iterations
                # ),
                # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
                record_shapes=True,  # Records tensor shapes for each op
                with_stack=True,
        ) as prof:

            for i in range(10):
                t = i * self.dt
                prof.step()
                self._solve_step(t)

        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        prof.export_chrome_trace("trace.json")

    @abstractmethod
    def _step(self, t):
        """
        Perform a single time step of the solver.

        Args:
            i: The index of the current time step.
        """
        pass

    def _euler_step(self, U, t):
        prim_a, _ = self.cells.convert_state_to_value(U)
        U_i_1 = U + self.dt * self.eq.forward(prim_a, self.dt, t)
        return U_i_1

    def _forward_state(self, U, t):
        prim, _ = self.cells.convert_state_to_value(U)
        return self.eq.forward(prim, self.dt, t)
