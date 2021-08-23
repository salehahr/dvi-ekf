from configuration import Config
from Filter import Filter
from tqdm import tqdm

## initialise objects
config      = Config()
camera      = config.get_camera()
imu         = config.get_imu(camera, gen_ref=True)
x0, cov0    = config.get_IC(imu, camera)
kf          = Filter(config, imu, x0, cov0)

## filter main loop (t>=1)
config.print_config()
old_t           = config.min_t
cap_t           = config.cap_t
cam_timestamps  = tqdm(enumerate(camera.t[1:]),
                    total=camera.max_vals, initial=1)

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

## plot results
kf.plot(config, t, camera.traj)