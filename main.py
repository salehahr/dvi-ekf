import numpy as np
np.set_printoptions(precision=1, linewidth=150)
import quaternion

from Filter import States, Filter
from Trajectory import VisualTraj, ImuTraj

# load data
vis_traj = VisualTraj("mono", "./trajs/offline_mandala0_mono.txt")
imu_traj = ImuTraj(filepath="./trajs/mandala0_imu.txt",
        vis_data=vis_traj, num_imu_between_frames=100)

# initial states
p0 = [vis_traj.x[0], vis_traj.y[0], vis_traj.z[0]]
v0 = [0., 0., 0.]
q0 = np.quaternion(vis_traj.qw[0],
        vis_traj.qx[0], vis_traj.qy[0], vis_traj.qz[0])
bw0 = [0., 0., 0.]
ba0 = [0., 0., 0.]
scale0 = 1.
p_BC_0 = [0., 0., 0.]
q_BC_0 = np.quaternion(1, 0, 0, 0)

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

# filter main loop
t_start = vis_traj.t[0]
t_end = vis_traj.t[-1]

for i, t in enumerate(vis_traj.t):
    print(f"i: {i}, camera time: {t}")
    current_vis = vis_traj.at_index(i)

    # initialisation
    if i == 0:
        x0 = States(p0, v0, q0, bw0, ba0, scale0, p_BC_0, q_BC_0)
        cov0 = np.square(np.diag(stdevs0))

        num_states = x0.size
        num_meas = 7
        num_control = len(u0)

        kf = Filter(num_states, num_meas, num_control)
        kf.set_states(x0)
        kf.set_covariance(cov0)

        old_t = t

        continue

    old_t = t