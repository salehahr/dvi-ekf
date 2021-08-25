from configuration import Config
from tqdm import tqdm, trange
import copy

## initialise objects
config          = Config()
kf, camera, imu = config.init_filter_objects()
x0, cov0        = config.get_IC(imu, camera)

## simulation runs
num_kf_runs, cap_t      = config.num_kf_runs, config.cap_t
dof_mses, dof_mse_best  = [], 1e10

for k in range(num_kf_runs):
    run_id = k + 1
    run_desc_str = f'KF run {run_id}/{num_kf_runs}'
    kf.run(camera, config.real_joint_dofs, run_id, run_desc_str)

    # save run and mse
    dof_mses.append(kf.dof_metric)
    if kf.dof_metric < dof_mse_best:
        dof_mse_best = kf.dof_metric
        kf_best = copy.deepcopy(kf)

    # reset for next run
    kf.reset(config, x0, cov0)

## results
dof_mse = sum(dof_mses) / len(dof_mses)
print(f'Best run: #{kf_best.run_id}; average MSE = {dof_mse:.2E}')
kf_best.save(config)
kf_best.plot(config, camera.traj, compact=True)