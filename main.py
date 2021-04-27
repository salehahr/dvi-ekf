import numpy as np
np.set_printoptions(precision=1, linewidth=150)
import matplotlib.pyplot as plt

from Filter import States, Filter, VisualTraj

# load data
from generate_data import mono_traj, stereoGT_traj, imu_gt_traj, min_t, max_t

imu_traj = imu_gt_traj
# imu_traj = imu_gt_traj.noisy

# initial states
p0 = [mono_traj.x[0], mono_traj.y[0], mono_traj.z[0]]
v0 = [0., 0., 0.]
q0 = [mono_traj.qx[0], mono_traj.qy[0], mono_traj.qz[0],
        mono_traj.qw[0]]
bw0 = [0., 0., 0.]
ba0 = [0., 0., 0.]
scale0 = 1.
p_BC_0 = [0., 0., 0.]
q_BC_0 = [0, 0, 0, 1]

# initial input
na0 = [0.01, 0.01, 0.01]
nba0 = [0.01, 0.01, 0.01]
nw0 = [0.01, 0.01, 0.01]
nbw0 = [0.01, 0.01, 0.01]
u0 = np.hstack((na0, nba0, nw0, nbw0))

# initial covariances
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]
stdev_bw = [0.1, 0.1, 0.1]
stdev_ba = [0.1, 0.1, 0.1]
stdev_scale = 0.4
stdev_p_BC = [0.1, 0.1, 0.1]
stdev_q_BC = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q, \
        stdev_bw, stdev_ba, stdev_scale, stdev_p_BC, stdev_q_BC))

# measurement noise
R_p = [0.01*0.01 for i in range(3)]
R_q = [0.01 for i in range(3)]
R = np.diag(np.hstack((R_p, R_q)))

# filter main loop
t_start = mono_traj.t[0]
t_end = mono_traj.t[-1]

traj = VisualTraj("kf")
for i, t in enumerate(mono_traj.t):
    current_vis = mono_traj.at_index(i)

    # initialisation
    if i == 0:
        x0 = States(p0, v0, q0, bw0, ba0, scale0, p_BC_0, q_BC_0)
        cov0 = np.square(np.diag(stdevs0))

        num_states = x0.size
        num_meas = 7
        num_control = len(u0)

        kf = Filter(num_states, num_meas, num_control)
        kf.dt = 0.
        kf.states = x0
        kf.set_covariance(cov0)

        current_imu = imu_traj.at_index(i)
        kf.om_old = current_imu.om
        kf.acc_old = current_imu.acc

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
            kf.propagate_covariance(current_imu)
            
            traj.append_state(ti, kf.states)

            old_ti = ti

    # update
    # kf.update(current_vis, R)

    old_t = t

# plot
axes = None
axes = stereoGT_traj.plot(axes)
axes = mono_traj.plot(axes)
axes = traj.plot(axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()