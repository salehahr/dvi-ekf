from configuration import Config
from Filter import Filter

## initialise objects
config      = Config(__file__)
camera      = config.get_camera()
imu         = config.get_imu(camera, gen_ref=True)
x0, cov0    = config.get_IC(imu, camera)
kf          = Filter(config, imu, x0, cov0)

## filter main loop (t>=1)
config.print_config()
old_t = config.min_t
cap_t = config.cap_t

for i, t in enumerate(camera.t[1:]):

    # propagate
    # simulate imu queue
    queue = imu.cam.generate_queue(old_t, t)
    old_ti = old_t

    print(f"Predicting... t={queue.t[0]}")
    for ii, ti in enumerate(queue.t):
        interp = queue.at_index(ii)
        om, acc = imu.eval_expr_single(ti, *config.real_joint_dofs,
            interp.acc, interp.R,
            interp.om, interp.alp, )
        imu.ref.append_value(ti, interp.vec)

        kf.dt = ti - old_ti
        kf.propagate(ti, om, acc)

        old_ti = ti

    # update
    if not config.do_prop_only:
        current_vis = camera.traj.at_index(i + 1) # not counting IC
        K = kf.update(current_vis)
        if K is None: break

    old_t = t

    # capping of simulation data
    if cap_t is not None and t >= cap_t: break

## plot results
from plotter import plot_trajectories
plot_trajectories(config, t, kf.traj, camera, imu)