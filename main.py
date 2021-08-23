from configuration import Config
from Filter import Filter
from tqdm import tqdm, trange
import copy

## initialise objects
config      = Config()
camera      = config.get_camera()
imu         = config.get_imu(camera, gen_ref=True)
x0, cov0    = config.get_IC(imu, camera)
kf          = Filter(config, imu, x0, cov0)
cap_t       = config.cap_t
num_kf_runs = config.num_kf_runs

dof_metric, dof_mse_best = 0, 1000
for k in range(num_kf_runs):

    ## filter main loop (t>=1)
    old_t           = config.min_t
    cam_timestamps  = tqdm(enumerate(camera.t[1:]),
                        total=camera.max_vals, initial=1,
                        desc=f'KF run {k+1}/{num_kf_runs}',
                        dynamic_ncols=True, postfix={'MSE': ''})
    kf.run_id       = k + 1

    for i, t in cam_timestamps:
        # propagate
        kf.propagate_imu(old_t, t, config.real_joint_dofs)

        # update
        if not config.do_prop_only:
            current_cam = camera.at_index(i + 1) # not counting IC
            K = kf.update(t, current_cam)
            if K is None: break

        kf.calculate_metric(config.real_joint_dofs)

        # capping of simulation data
        if cap_t is not None and t >= cap_t: break

        old_t = t
        cam_timestamps.set_postfix({'MSE': f'{kf.dof_metric:.2E}'})

    # save best run
    if kf.dof_metric < dof_mse_best: kf_best = copy.deepcopy(kf)

    # reset for next run
    dof_metric += kf.dof_metric
    kf.reset(config, x0, cov0)


## results
dof_metric = dof_metric / (num_kf_runs * i * 6) # runs * meas * len(dofs)
print(f'Best run: #{kf_best.run_id}; average MSE = {dof_metric:.2E}')
kf_best.plot(config, t, camera.traj)