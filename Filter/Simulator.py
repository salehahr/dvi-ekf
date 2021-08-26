import copy
from tqdm import trange
from scipy.optimize import basinhopping
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
        self._kp = config.scale_process_noise
        self._km = config.scale_meas_noise
        self._rwp = config.stdev_dofs_p
        self._rwr = config.stdev_dofs_r

        # simulation run params
        self.num_kf_runs = config.num_kf_runs
        self.cap_t      = config.cap_t
        self.run_progress = True

        # results
        self.dof_mses = []
        self.dof_mse_best  = 1e10
        self.dof_mse_avg = None
        self.kf_best = None
        self.best_x = None

    # optimisation variables
    @property
    def kp(self):
        return self._kp
    @kp.setter
    def kp(self, val):
        self._kp = val
        self.config.scale_process_noise = val

    @property
    def km(self):
        return self._km
    @kp.setter
    def km(self, val):
        self._km = val
        self.config.scale_meas_noise = val

    @property
    def rwp(self):
        return self._rwp
    @kp.setter
    def rwp(self, val):
        self._rwp = val
        self.config.stdev_dofs_p = val

    @property
    def rwr(self):
        return self._rwr
    @kp.setter
    def rwr(self, val):
        self._rwr = val
        self.config.stdev_dofs_r = val

    @property
    def best_run_id(self):
        return self.kf_best.run_id

    def run(self, disp_config=False, save_best=False):
        self.kf.config = self.config

        if disp_config: self.config.print_config()

        run_bar = trange(self.num_kf_runs,
                    desc='KF runs',
                    postfix={'MSE': ''},
                    disable = not self.run_progress)
        self.dof_mses = []
        for k in run_bar:
            run_id = k + 1
            run_desc_str = f'KF run {run_id}/{self.num_kf_runs}'

            self.kf.run(self.camera, run_id, run_desc_str)

            # save run and mse
            self.dof_mses.append(self.kf.dof_metric)
            if self.kf.dof_metric < self.dof_mse_best:
                self.dof_mse_best = self.kf.dof_metric

                if save_best:
                    self.kf_best = copy.deepcopy(self.kf)

            # reset for next run
            self.kf.reset(self.x0, self.cov0)
            run_bar.set_postfix({'dof_mse_best': f'{self.dof_mse_best:.2E}'})

        self.dof_mse_avg = sum(self.dof_mses) / len(self.dof_mses)

    def optimise(self):
        def optim_func(x):
            self.kp, self.km, self.rwp, self.rwr = x
            self.run()
            return self.dof_mse_avg

        def print_fun(x, f, accepted):
            print(f"for x {x}: average MSE {f:.4f} accepted {int(accepted)}")

        kp0 = self.config.scale_process_noise
        km0 = self.config.scale_meas_noise
        rwp0 = self.config.stdev_dofs_p
        rwr0 = self.config.stdev_dofs_r
        x0 = [kp0, km0, rwp0, rwr0]

        # global minimum: kp = 4.024E-01, mse = 4.996
        bnds = ((0, None), (0, None), (0, 3), (0, np.deg2rad(10)))
        minimizer_kwargs = {'method'    : 'L-BFGS-B',
                            'bounds'    : bnds}

        self.run_progress = False
        self.kf.progress_bar = False

        print("Initial config.")
        self.config.print_config()
        ret = basinhopping(optim_func, x0,
                            minimizer_kwargs=minimizer_kwargs,
                            callback=print_fun,
                            niter=1)

        # results
        print(f"\nglobal minimum using params: {ret.x},\nmse = {ret.fun:.3f}")
        self.save_params(ret.x)
        self.dof_mse = ret.fun

    def save_params(self, x, filename=None):
        # 7.779E-09 1.355 0.544 0.034 average MSE 5.0474 (fast run)
        filename = './opt-tune-params.txt' if not filename else filename
        with open(filename, 'w+') as f:
            f.write(x)

    def save_and_plot(self):
        print(f'Best run: #{self.best_run_id}; average MSE = {self.dof_mse_avg:.2E}')
        self.kf_best.save()
        self.kf_best.plot(self.camera.traj, compact=True)