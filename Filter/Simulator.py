import copy

class Simulator(object):
    def __init__(self, config):
        # simulation objects
        self.config = config
        _, camera, imu = config.init_filter_objects()
        self.x0, self.cov0        = config.get_IC(imu, camera)

        # simulation run params
        self.num_kf_runs = config.num_kf_runs
        self.cap_t      = config.cap_t
        
        # results
        self.dof_mses = []
        self.dof_mse_best  = 1e10
        self.kf_best = None

    @property
    def dof_mse(self):
        return sum(self.dof_mses) / len(self.dof_mses)

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

    def save_and_plot(self, camera):
        print(f'Best run: #{self.best_run_id}; average MSE = {self.dof_mse:.2E}')
        self.kf_best.save(self.config)
        self.kf_best.plot(self.config, camera.traj, compact=True)