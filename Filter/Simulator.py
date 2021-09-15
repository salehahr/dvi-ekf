import copy
from tqdm import trange
from scipy.optimize import differential_evolution
import numpy as np

from Filter import Filter

class Simulator(object):
    def __init__(self, config):
        # simulation objects
        kf, camera, imu = config.init_filter_objects()
        self.config = config
        self.kf = kf
        self.camera = camera
        self.imu = imu

        self.x0, self.cov0        = config.get_IC(imu, camera)

        # optimisation variables
        # for now ignoring dofs
        self._optim_std = [*config.process_noise_rw_std,
                *config.meas_noise_std]

        # simulation run params
        self.num_kf_runs = config.num_kf_runs
        self.cap_t      = config.cap_t
        self.show_run_progress = True

        # results
        self.mses = []
        self.mse_best  = 1e10
        self.mse_avg = None
        self.kf_best = None
        self.best_x = None

    # optimisation variables
    # for now ignoring dofs
    @property
    def optim_std(self):
        return self._optim_std
    @optim_std.setter
    def optim_std(self, val):
        self._optim_std = val
        self.config.process_noise_rw_std = val[0:7]
        self.config.meas_noise_std = val[7:8]

        self.kf.update_noise_matrices()

    @property
    def best_run_id(self):
        return self.kf_best.run_id

    def run_once(self):
        self.kf.run(self.camera, 0, 'KF run', verbose=True)
        self.mse_best = self.kf.mse
        print(f'\t MSE: {self.mse_best:.2E}')

    def run(self, disp_config=False, save_best=False, verbose=True):
        """ Runs KF on the camera trajectory several times.
            Calculates the mean squared error of the DOFs,
                averaged from all the KF runs.
        """
        # make sure that KF has the right config
        if self.config.mode == 'tune':
            self.kf.config = self.config

        if disp_config: self.config.print_config()

        run_bar = trange(self.num_kf_runs,
                    desc='KF runs',
                    disable = not self.show_run_progress)

        self.mses = []
        for k in run_bar:
            run_id = k + 1
            run_desc_str = f'KF run {run_id}/{self.num_kf_runs}'

            self.kf.run(self.camera, run_id, run_desc_str, verbose)

            # save run and mse
            self.mses.append(self.kf.mse)
            if self.kf.mse < self.mse_best:
                self.mse_best = self.kf.mse

                if save_best:
                    self.kf_best = copy.deepcopy(self.kf)

            # reset for next run
            self.reset_kf()

        self.mse_avg = sum(self.mses) / len(self.mses)
        if verbose:
            print(f'\tOptimvars: {self.optim_std}')
            print(f'\tDOF MSE: {self.mse_avg:.2E}')

    def reset_kf(self):
        self.kf = Filter(self.config, self.imu, self.x0, self.cov0,
            self.config.frozen_dofs)

    def optimise(self):
        """ For tuning the KF parameters.
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

            res_str = [f"Current optim. variables:",
                f"{rwp} cm",
                f"{rwr_deg} deg",
                f"{notchdd_deg} deg",
                f'{pcam} cm',
                f'{rcam_deg} deg',
                f'{notch_deg} deg',
                f'MSE: {self.mse_avg}',
                f'Convergence: {convergence}\n\n']

            self.file.write('\n'.join(res_str))
            print(res_str)

        file = open('output.txt', 'a+')
        self.file = file

        bounds = (  
                    (0, 10), # random walk p
                    (0, 10), 
                    (0, 10), 
                    (0, np.deg2rad(5)), # random walk r
                    (0, np.deg2rad(5)),
                    (0, np.deg2rad(5)),
                    (0, np.deg2rad(5)),     # notchdd
                    (0, 0.2),     # pcam
                    (0, 0.2),
                    (0, 0.2),
                    (0, np.deg2rad(10)),     # rcam
                    (0, np.deg2rad(10)),
                    (0, np.deg2rad(10)),
                    (0, np.deg2rad(1))  # notch
                    )

        self.show_run_progress = False
        self.kf.show_progress = False

        print('Running optimiser (differential evolution)... ')
        print('Initial config')
        self.config.print_config()

        ret = differential_evolution(
                            optim_func, 
                            bounds,
                            strategy = 'best1bin',
                            maxiter = 1,
                            popsize = 1,
                            disp = True,
                            # x0 = x0, # not available in my python setup
                            callback = print_fun,
                            updating = 'immediate')

        # results
        print(f"\nglobal minimum using params: {ret.x},\nmse = {ret.fun:.3f}")
        self.save_params(ret.x, filename='./opt-tune-params-de.txt')
        self.dof_mse = ret.fun

        file.close()

    def save_params(self, x, filename=None):
        # 7.779E-09 1.355 0.544 0.034 average MSE 5.0474 (fast run)
        filename = './opt-tune-params.txt' if not filename else filename
        with open(filename, 'w+') as f:
            f.write(x)

    def save_and_plot(self, compact=True):
        if self.kf_best:
            self.kf_best.save()
            self.kf_best.plot(self.camera, compact=compact)
            print(f'Best run: #{self.best_run_id}; average MSE = {self.mse_avg:.2E}')
        else:
            self.kf.save()
            self.kf.plot(self.camera, compact=compact)