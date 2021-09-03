import copy
from tqdm import trange
from scipy.optimize import differential_evolution
import numpy as np

class Simulator(object):
    def __init__(self, config):
        # simulation objects
        kf, camera, imu = config.init_filter_objects()
        self.config = config
        self.kf = kf
        self.camera = camera

        self.x0, self.cov0        = config.get_IC(imu, camera)

        # optimisation variables
        self._q = config.meas_noise

        # simulation run params
        self.num_kf_runs = config.num_kf_runs
        self.cap_t      = config.cap_t
        self.show_run_progress = True

        # results
        self.dof_mses = []
        self.dof_mse_best  = 1e10
        self.dof_mse_avg = None
        self.kf_best = None
        self.best_x = None

    # optimisation variables

    @property
    def q(self):
        return self._q
    @q.setter
    def q(self, val):
        self._q = val
        self.config.meas_noise = val

    @property
    def best_run_id(self):
        return self.kf_best.run_id

    def run(self, disp_config=False, save_best=False, verbose=True):
        """ Runs KF on the camera trajectory several times.
            Calculates the mean squared error of the DOFs,
                averaged from all the KF runs.
        """
        # make sure that KF has the right config
        self.kf.config = self.config
        if disp_config: self.config.print_config()

        run_bar = trange(self.num_kf_runs,
                    desc='KF runs',
                    disable = not self.show_run_progress)

        self.dof_mses = []
        for k in run_bar:
            run_id = k + 1
            run_desc_str = f'KF run {run_id}/{self.num_kf_runs}'

            self.kf.run(self.camera, run_id, run_desc_str, verbose)

            # save run and mse
            self.dof_mses.append(self.kf.dof_metric)
            if self.kf.dof_metric < self.dof_mse_best:
                self.dof_mse_best = self.kf.dof_metric

                if save_best:
                    self.kf_best = copy.deepcopy(self.kf)

            # reset for next run
            self.kf.reset(self.x0, self.cov0)

        self.dof_mse_avg = sum(self.dof_mses) / len(self.dof_mses)
        if verbose:
            print(f'\tDOF MSE: {self.dof_mse_avg:.2E}')

    def optimise(self):
        """ For tuning the KF parameters.
            Currently only for kp (scale factor for process noise).
        """
        def optim_func(x):
            self.kp, self.km, self.rwp, self.rwr = x
            self.run(verbose=False)
            return self.dof_mse_avg

        def print_fun(x0, convergence):
            print(f"current param set: {x0}")

        def optim_func_kp_only(x):
            self.kp = x
            self.run(verbose=False)
            return self.dof_mse_avg

        bounds = ((0, 1), (0, 5), (0, 3), (0, np.deg2rad(10)))
        bounds_kp_only = ((1e-4, 1e-2),)

        self.show_run_progress = False
        self.kf.show_progress = False

        print('Running optimiser (differential evolution)... ')
        print('Initial config')
        self.config.print_config()

        ret = differential_evolution(
                            # optim_func, 
                            optim_func_kp_only, 
                            # bounds,
                            bounds_kp_only,
                            strategy = 'best1bin',
                            disp = True,
                            # x0 = x0,
                            callback = print_fun)

        print("Initial config.")
        self.config.print_config()

        # results
        print(f"\nglobal minimum using params: {ret.x},\nmse = {ret.fun:.3f}")
        self.save_params(ret.x, filename='./opt-tune-params-de.txt')
        self.dof_mse = ret.fun

    def save_params(self, x, filename=None):
        # 7.779E-09 1.355 0.544 0.034 average MSE 5.0474 (fast run)
        filename = './opt-tune-params.txt' if not filename else filename
        with open(filename, 'w+') as f:
            f.write(x)

    def save_and_plot(self, compact=True):
        print(f'Best run: #{self.best_run_id}; average MSE = {self.dof_mse_avg:.2E}')
        self.kf_best.save()
        self.kf_best.plot(self.camera, compact=compact)