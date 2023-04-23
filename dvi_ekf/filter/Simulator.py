from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.optimize import differential_evolution
from tqdm import trange

from dvi_ekf.models import create_camera
from dvi_ekf.models.Imu import Imu
from dvi_ekf.models.Probe import SimpleProbe, SymProbe
from dvi_ekf.tools.files import save_tuned_params
from dvi_ekf.tools.utils import get_ic

from .Filter import Filter

if TYPE_CHECKING:
    from dvi_ekf.filter.state import State
    from dvi_ekf.models.Camera import Camera
    from dvi_ekf.tools.config import Config, SimMode


class Simulator(object):
    """
    Simulation object with simulation settings as attributes.
    Provides methods for running and tuning the Kalman filter simulation.
    """

    def __init__(self, config: Config):
        # probe must be defined first as the IMU depends on the GT joint values
        probe = SimpleProbe(
            scope_length=config.model.length, theta_cam=config.model.angle
        )
        self.probe = probe
        self.sym_probe = SymProbe(probe)

        # initialise simulation objects
        self._config: Optional[Config] = None
        self.camera: Optional[Camera] = None
        self.imu: Optional[Imu] = None

        self.x0: Optional[State] = None
        self._cov0: Optional[np.ndarray] = None

        config.update_dofs(probe)
        self._update_config(config)

        self.kf = Filter(self)

        # optimisation variables -- for now ignoring dofs
        self._optim_std: List[float] = [
            *self.config.process_noise_rw_std,
            *self.config.meas_noise_std,
        ]

        # simulation run params
        self.mode: SimMode = config.sim.mode
        self.num_kf_runs: int = config.sim.num_kf_runs
        self.show_run_progress: bool = True

        # results
        self.mses: List[float] = []
        self.mse_best: float = 1e10
        self.mse_avg: Optional[float] = None
        self._kf_best: Optional[Filter] = None
        self._dof_mse: Optional[float] = None
        self._file = None

    # optimisation variables
    # for now ignoring dofs
    @property
    def optim_std(self) -> List[float]:
        return self._optim_std

    @optim_std.setter
    def optim_std(self, val: List[float]) -> None:
        self._optim_std = val

        # set the config values
        # -- note: random walk might be problematic due to
        # division by interframe values in the initial definition
        self.config.process_noise_rw_std = val[0:7]
        self.config.meas_noise_std = val[7:8]

        self.kf.update_noise_matrices()

    @property
    def config(self):
        return self._config

    @property
    def best_run_id(self) -> int:
        return self._kf_best.run_id

    @property
    def cov0(self) -> np.ndarray:
        """Returns a copy of the initial covariance matrix."""
        return self._cov0.copy()

    def _update_config(self, new_config: Config) -> None:
        """
        Creates new camera and IMU objects from the new config.
        Affects the initial states.
        :param new_config: updated config object
        """
        self._config = new_config

        self.camera = create_camera(new_config)
        self.imu = Imu.create(new_config.imu, self.camera, self.probe, gen_ref=True)

        self.x0 = get_ic(self.camera, self.imu, new_config.ic_imu_dofs)
        self._cov0 = new_config.cov0_matrix

    def run_once(self) -> None:
        """Only performs a single run of the filter."""
        self.kf.run(self.camera, 0, "KF run")
        self.mse_best = self.kf.mse
        print(f"\t MSE: {self.mse_best:.2E}")

    def run(self, disp_config=False, save_best=False, verbose=True) -> None:
        """Runs KF on the camera trajectory several times.
        Calculates the mean squared error of the DOFs,
            averaged from all the KF runs.
        """
        # make sure that KF has the right config
        if self.mode == "tune":
            self.kf.config = self.config

        if disp_config:
            self.config.print()

        run_bar = trange(
            self.num_kf_runs, desc="KF runs", disable=not self.show_run_progress
        )

        self.mses = []
        for k in run_bar:
            run_id = k + 1
            run_desc_str = f"KF run {run_id}/{self.num_kf_runs}"

            self.kf.run(self.camera, run_id, run_desc_str)

            # save run and mse
            self.mses.append(self.kf.mse)
            if self.kf.mse < self.mse_best:
                self.mse_best = self.kf.mse

                if save_best:
                    self._kf_best = copy.deepcopy(self.kf)

            # reset for next run
            self.reset_kf()

        self.mse_avg = sum(self.mses) / len(self.mses)
        if verbose:
            print(f"\tOptimvars: {self.optim_std}")
            print(f"\tDOF MSE: {self.mse_avg:.2E}")

    def reset_kf(self) -> None:
        self.kf = Filter(self)

    def optimise(self) -> None:
        """For tuning the KF parameters.
        Currently only for kp (scale factor for process noise).
        """

        def optim_func(x):
            self.optim_std = x
            self.run(verbose=False)
            return self.mse_avg

        def print_fun(x0, convergence):
            rwp = x0[0:3]
            rwr = x0[3:6]
            notchdd = x0[6]

            pcam = x0[7:10]
            rcam = x0[10:13]
            notch = x0[13]

            rwr_deg = np.rad2deg(rwr)
            notchdd_deg = np.rad2deg(notchdd)
            rcam_deg = np.rad2deg(rcam)
            notch_deg = np.rad2deg(notch)

            res_str = [
                f"Current optim. variables:",
                f"{rwp} cm",
                f"{rwr_deg} deg",
                f"{notchdd_deg} deg",
                f"{pcam} cm",
                f"{rcam_deg} deg",
                f"{notch_deg} deg",
                f"MSE: {self.mse_avg}",
                f"Convergence: {convergence}\n\n",
            ]

            self._file.write("\n".join(res_str))
            print(res_str)

        self._file = open("output.txt", "a+")

        bounds = (
            (0, 10),  # random walk p
            (0, 10),
            (0, 10),
            (0, np.deg2rad(5)),  # random walk r
            (0, np.deg2rad(5)),
            (0, np.deg2rad(5)),
            (0, np.deg2rad(5)),  # notchdd
            (0, 0.2),  # pcam
            (0, 0.2),
            (0, 0.2),
            (0, np.deg2rad(10)),  # rcam
            (0, np.deg2rad(10)),
            (0, np.deg2rad(10)),
            (0, np.deg2rad(1)),  # notch
        )

        self.show_run_progress = False
        self.kf.show_progress = False

        print("Running optimiser (differential evolution)... ")
        print("Initial config")
        self.config.print()

        ret = differential_evolution(
            optim_func,
            bounds,
            strategy="best1bin",
            maxiter=1,
            popsize=1,
            disp=True,
            # x0 = x0, # not available in my python setup
            callback=print_fun,
            updating="immediate",
        )

        # results
        print(f"\nglobal minimum using params: {ret.x},\nmse = {ret.fun:.3f}")
        save_tuned_params(ret.x, filename="./opt-tune-params-de.txt")
        self._dof_mse = ret.fun

        self._file.close()

    def plot(self, compact=True) -> None:
        if self._kf_best:
            self._kf_best.plot(self.camera, compact=compact)
            print(f"Best run: #{self.best_run_id}; average MSE = {self.mse_avg:.2E}")
        else:
            self.kf.plot(self.camera, compact=compact)
