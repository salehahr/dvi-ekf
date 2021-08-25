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

    ## filter main loop (t>=1)
    old_t           = config.min_t
    cam_timestamps  = tqdm(enumerate(camera.t[1:]),
                        total=camera.max_vals, initial=1,
                        desc=f'KF run {k+1}/{num_kf_runs}',
                        dynamic_ncols=True, postfix={'MSE': ''})
    kf.run_id       = k + 1

    for i, t in cam_timestamps:
        i_cam = i + 1 # not counting IC
        kf.run_one_epoch(old_t, t, i_cam, camera, config.real_joint_dofs)

        old_t = t
        cam_timestamps.set_postfix({'sum error': f'{kf.dof_metric:.2E}'})

    # save run
    kf.dof_metric = kf.dof_metric / (i * 6) # normalise
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
kf_best.plot(config, t, camera.traj, compact=True)