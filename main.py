import sys

import numpy as np
np.set_printoptions(precision=1, linewidth=150)

import matplotlib.pyplot as plt
from Filter import States, Filter, VisualTraj

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

    return kf

def plot_savefig(fig, figname):
    print(f"Saving file \"{figname}\". ")
    fig.savefig(figname)

def plot_trajectories():
    axes = stereoGT_traj.plot()
    if not do_prop_only:
        axes = mono_traj.plot(axes)
    axes = kf.traj.plot(axes, min_t=min_t, max_t=max_t)

def plot_velocities():
    if do_plot_vel:
        v_axes = stereoGT_traj.plot_velocities()
        v_axes = kf.traj.plot_velocities(v_axes, min_t=min_t, max_t=max_t)

def plot_noise_sensitivity(Q, Rp, Rq):
    """ plots sensitivity to measurement noise R and process noise Q """
    R_axes = stereoGT_traj.plot_sens_noise(Rp, Rq, Q)
    R_axes = kf.traj.plot_sens_noise(Rp, Rq, Q, R_axes,
            min_t=min_t, max_t=max_t)

    figname = f"./img/Rp{Rp}_Rq{Rqval}_Q{Q}.png"
    plot_savefig(R_axes[0].figure, figname)

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

# filter main loop
for i, t in enumerate(mono_traj.t):
    current_vis = mono_traj.at_index(i)

    # propagate
    imu_queue = imu_traj.get_queue(old_t, t)
    if imu_queue:
        for ii, ti in enumerate(imu_queue.t):
            current_imu = imu_queue.at_index(ii)
            kf.dt = ti - old_ti

            kf.propagate(ti, current_imu, Qc)

            old_ti = ti

    # update
    if not do_prop_only:
        kf.update(current_vis, R)

    old_t = t

# plots
plot_trajectories()
plot_velocities()
plot_noise_sensitivity(Qval, Rpval, Rqval)

plt.legend()
plt.show()