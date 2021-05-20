import sys

import numpy as np
np.set_printoptions(precision=1, linewidth=150)

import matplotlib.pyplot as plt
from Filter import States, Filter, VisualTraj, MatrixPlotter

from plotter import plot_savefig, plot_trajectories, plot_velocities, plot_noise_sensitivity

def parse_arguments():
    def print_usage():
        print(f"Usage: {__file__} <prop> <noise> [<vel>]")
        print("\t <prop>  - prop / all")
        print("\t <noise> - noise / nonoise")
        print("Optional arguments:")
        print("\t <vel>   - vel")
        sys.exit()

    try:
        do_prop_only = False or (sys.argv[1] == 'prop')
        use_noisy_imu = False or (sys.argv[2] == 'noise')
    except:
        print_usage()

    try:
        do_plot_vel =  (sys.argv[3] == 'vel')
    except:
        do_plot_vel = False

    return do_prop_only, use_noisy_imu, do_plot_vel
do_prop_only, use_noisy_imu, do_plot_vel = parse_arguments()

def gen_noise_matrices(Q, Rp, Rq):
    # process noise
    stdev_na = [Q] * 3
    stdev_nw = stdev_na
    stdevs = np.hstack((stdev_na,  stdev_nw))
    Qc = np.square(np.diag(stdevs))

    # measurement noise
    Rp = [Rp] * 3
    Rq = [Rq] * 4
    R = np.diag(np.hstack((Rp, Rq)))

    return Qc, R

# load data
from generate_data import mono_traj, stereoGT_traj, imu_gt_traj
from generate_data import IC, cov0, min_t, max_t
imu_traj = (imu_gt_traj.noisy if use_noisy_imu else imu_gt_traj)

def init_kf(current_imu):
    num_meas, num_control = 7, 12
    kf = Filter(IC, cov0, num_meas, num_control)
    kf.om_old, kf.acc_old = current_imu.om, current_imu.acc

    # debugging
    kf.P_mp = MatrixPlotter("P", min_t, kf.P, max_row=6, max_col=6)

    return kf

# noise values
Qval = 1e-3
Rpval = 1e3
Rqval = 0.05
Qc, R = gen_noise_matrices(Qval, Rpval, Rqval)

# initialisation
current_imu = imu_traj.at_index(0)
kf = init_kf(current_imu)

old_t = min_t

# filter main loop -- start at 1 b/c we have initial values
for i, t in enumerate(imu_traj.t[1:]):

    current_imu = imu_traj.at_index(i)
    kf.dt = t - old_t
    
    kf.propagate(t, current_imu, Qc)
    # for plotting matrices
    kf.P_mp.append(t, kf.P)
    
    if not do_prop_only:
        current_vis = mono_traj.get_meas(old_t, t)

        if current_vis:
            kf.update(current_vis, R)

    old_t = t

# plots
plot_trajectories(kf.traj, do_prop_only)
# plot_velocities(kf.traj, do_plot_vel)
# plot_noise_sensitivity(kf.traj, Qval, Rpval, Rqval)

P_mp_axes = kf.P_mp.plot(min_t=min_t, max_t=max_t)

plt.legend()
plt.show()