import numpy as np
np.set_printoptions(precision=1, linewidth=150)

import matplotlib.pyplot as plt
from Filter import States, Filter, VisualTraj

import sys

# parse arguments
do_prop_only = False
do_plot_vel = False
use_noisy_imu = False
if len(sys.argv) == 1:
    print(f"Usage: {__file__} <prop> <noise> [<vel>]")
    print("\t <prop>  - prop / all")
    print("\t <noise> - noise / nonoise")
    print("Optional arguments:")
    print("\t <vel>   - vel")
    sys.exit()
if len(sys.argv) > 1:
    do_prop_only = (sys.argv[1] == 'prop')
if len(sys.argv) > 2:
    use_noisy_imu = (sys.argv[2] == 'noise')
if len(sys.argv) > 3:
    do_plot_vel = (sys.argv[3] == 'vel')

# load data
from generate_data import mono_traj, stereoGT_traj, imu_gt_traj
from generate_data import IC, cov0, min_t, max_t

if use_noisy_imu:
    imu_traj = imu_gt_traj.noisy
else:
    imu_traj = imu_gt_traj

# noise values
Qval = 1e-3
Rpval = 0.1
Rqval = 0.05

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

# filter main loop
kf_traj = VisualTraj("kf")

for i, t in enumerate(mono_traj.t):
    current_vis = mono_traj.at_index(i)

    # initialisation
    if i == 0:
        num_states, num_meas, num_control = IC.size, 7, 12
        kf = Filter(num_states, num_meas, num_control)

        kf.dt = 0.
        kf.states = IC
        kf.set_covariance(cov0)

        current_imu = imu_traj.at_index(i)
        kf.om_old, kf.acc_old = current_imu.om, current_imu.acc

        old_t = t
        old_ti = t
        
        continue

    # propagate
    imu_queue = imu_traj.get_queue(old_t, t)
    if imu_queue:
        for ii, ti in enumerate(imu_queue.t):
            current_imu = imu_queue.at_index(ii)
            kf.dt = ti - old_ti

            kf.propagate_states(current_imu)
            kf.propagate_covariance(current_imu, Qc)
            
            kf_traj.append_state(ti, kf.states)

            old_ti = ti

    # update
    if not do_prop_only:
        kf.update(current_vis, R)

    old_t = t

# plot
axes = None
axes = stereoGT_traj.plot(axes)
if not do_prop_only:
    axes = mono_traj.plot(axes)
axes = kf_traj.plot(axes, min_t=min_t, max_t=max_t)

# plot velocities
if do_plot_vel:
    v_axes = None
    v_axes = stereoGT_traj.plot_velocities(v_axes)
    v_axes = kf_traj.plot_velocities(v_axes, min_t=min_t, max_t=max_t)

# plot sensitivity to measurement noise R
R_axes = None
R_axes = stereoGT_traj.plot_sens_noise(Rpval, Rqval, Qval, R_axes)
if not do_prop_only:
    R_axes = mono_traj.plot_sens_noise(Rpval, Rqval, Qval, R_axes)
R_axes = kf_traj.plot_sens_noise(Rpval, Rqval, Qval, R_axes, min_t=min_t, max_t=max_t)
figname = f"./img/Rp{Rpval}_Rq{Rqval}_Q{Qval}.png"
print(figname)
R_axes[0].figure.savefig(figname)

plt.legend()
plt.show()