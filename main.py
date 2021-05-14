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
    stdev_na = [Qval] * 3
    stdev_nba = stdev_na
    stdev_nw = stdev_na
    stdev_nbw = stdev_na
    stdevs = np.hstack((stdev_na, stdev_nba, \
                stdev_nw, stdev_nbw))
    Qc = np.square(np.diag(stdevs))

    # measurement noise
    Rp = [Rpval] * 3
    Rq = [Rqval] * 3
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
    kf.R_WB_mp = MatrixPlotter("R_WB", min_t, kf.states.q.rot)
    kf.P_mp = MatrixPlotter("P", min_t, kf.P, max_row=3, max_col=3)

    return kf

# noise values
Qval = 1e-3
Rpval = 0.1
Rqval = 0.05
Qc, R = gen_noise_matrices(Qval, Rpval, Rqval)

# initialisation
current_imu = imu_traj.at_index(0)
kf = init_kf(current_imu)

old_t = min_t
old_ti = min_t

# filter main loop -- start at 1 b/c we have initial values
for i, t in enumerate(imu_traj.t[1:]):

    current_imu = imu_traj.at_index(i)
    kf.dt = t - old_t
    
    kf.propagate(t, current_imu, Qc)
    
    old_t = t

    # current_vis = mono_traj.at_index(i)

    # # propagate
    # imu_queue = imu_traj.get_queue(old_t, t)
    # if imu_queue:
        # for ii, ti in enumerate(imu_queue.t):

            # # for plotting matrices
            # kf.R_WB_mp.append(ti, kf.states.q.rot)
            # kf.P_mp.append(ti, kf.P)

            # old_ti = ti

    # # update
    # if not do_prop_only:
        # kf.update(current_vis, R)

    # old_t = t

# plots
plot_trajectories(kf.traj, do_prop_only)
# plot_velocities(kf.traj, do_plot_vel)
# plot_noise_sensitivity(kf.traj, Qval, Rpval, Rqval)

# R_WB_mp_axes = kf.R_WB_mp.plot(min_t=min_t, max_t=max_t)
# for ax in R_WB_mp_axes.reshape(-1):
    # ax.set_ylim(bottom=-1.1, top=1.1)
# P_mp_axes = kf.P_mp.plot(min_t=min_t, max_t=max_t)

plt.legend()
plt.show()