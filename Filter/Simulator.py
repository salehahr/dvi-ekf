import copy
from scipy.optimize import basinhopping

class Simulator(object):
    def __init__(self, config):
        # simulation objects
        self.config = config
        _, camera, imu = config.init_filter_objects()
        self.x0, self.cov0        = config.get_IC(imu, camera)

        # optim params
        self._kp = config.scale_process_noise

        # simulation run params
        self.num_kf_runs = config.num_kf_runs
        self.cap_t      = config.cap_t

        # results
        self.dof_mses = []
        self.dof_mse_best  = 1e10
        self.dof_mse_avg = None
        self.kf_best = None

    @property
    def kp(self):
        return self._kp
    @kp.setter
    def kp(self, val):
        self._kp = val
        self.config.scale_process_noise = val

    @property
    def best_run_id(self):
        return self.kf_best.run_id

    def run(self, kf, camera):
        for k in range(self.num_kf_runs):
            run_id = k + 1
            run_desc_str = f'KF run {run_id}/{self.num_kf_runs}'

            kf.run(camera, self.config.real_joint_dofs,
                run_id, run_desc_str)

            # save run and mse
            self.dof_mses.append(kf.dof_metric)
            if kf.dof_metric < self.dof_mse_best:
                self.dof_mse_best = kf.dof_metric
                self.kf_best = copy.deepcopy(kf)

            # reset for next run
            kf.reset(self.config, self.x0, self.cov0)

        self.dof_mse_avg = sum(self.dof_mses) / len(self.dof_mses)
        self.dof_mses = [] # clear

    def optimise(self, kf ,camera):
        def optim_func(kp):
            self.kp = kp
            self.run(kf, camera)
            return self.dof_mse_avg

        def print_fun(x, f, accepted):
            print(f"for kp {x[0]:.3E}: average MSE {f:.4f} accepted {int(accepted)}")

        kp0 = 0.0010
        # global minimum: kp = 4.024E-01, mse = 4.996
        minimizer_kwargs = {"method": "BFGS"}

        ret = basinhopping(optim_func, kp0,
                            minimizer_kwargs=minimizer_kwargs,
                            callback=print_fun,
                            niter=1)

        # results
        print(f"global minimum: kp = {ret.x[0]:.3E}, mse = {ret.fun:.3f}")
        self.best_kp = ret.x
        self.dof_mse = ret.fun

    def save_and_plot(self, camera):
        print(f'Best run: #{self.best_run_id}; average MSE = {self.dof_mse:.2E}')
        self.kf_best.save(self.config)
        self.kf_best.plot(self.config, camera.traj, compact=True)