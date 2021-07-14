from Filter import States, Filter, VisualTraj, ImuDesTraj, MatrixPlotter
import numpy as np

def parse_arguments():
    import sys

    def print_usage():
        print(f"Usage: {__file__} <prop> <noise>")
        print("\t <prop>  - prop / all")
        print("\t <noise> - noise / nonoise")
        sys.exit()

    try:
        do_prop_only = False or (sys.argv[1] == 'prop')
        use_noisy_imu = False or (sys.argv[2] == 'noise')
    except IndexError:
        print_usage()

    return do_prop_only, use_noisy_imu
do_prop_only, use_noisy_imu = parse_arguments()

# load data
from generate_data import probe_BtoC, cam, cam_interp, imu
from generate_data import IC, cov0, min_t, max_t, gen_noise_matrices

# noise values
Qval = 1e-3
Rpval, Rqval = 1e3, 0.05

# initialisation (t=0): IC, IMU buffer, noise matrices
kf = Filter(imu, IC, cov0)
kf.traj.append_state(cam.t[0], kf.states)
kf.Qc, kf.R = gen_noise_matrices(Qval, Rpval, Rqval)

# desired trajectory
imu_des = ImuDesTraj("imu soll", imu)

# filter main loop (t>=1)
old_t = min_t
for i, t in enumerate(cam.t[1:]):

    # propagate
    # queue = imu.traj.get_queue(old_t, t) # real imu data
    # simulate imu queue
    queue = cam_interp.generate_queue(old_t, t)
    old_ti = old_t

    print(f"Predicting... t={queue.t[0]}")
    for i, ti in enumerate(queue.t):
        interp = queue.at_index(i)
        om, acc = imu.eval_expr_single(ti, *probe_BtoC.joint_dofs,
            interp.acc, interp.R,
            interp.om, interp.alp, )
        imu_des.append_value(ti, interp)

        kf.dt = ti - old_ti
        kf.propagate(ti, om, acc)

        old_ti = ti

    # update
    if not do_prop_only:
        current_vis = cam.traj.at_index(i)
        kf.update(current_vis)

    old_t = t



# plots
from plotter import plot_trajectories
traj_name = 'mono'
plot_trajectories(kf.traj, traj_name, imu_des)