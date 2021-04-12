import numpy as np
np.set_printoptions(precision=1, linewidth=150)
import quaternion
import matplotlib.pyplot as plt

from Filter import States, Filter
from Trajectory import Trajectory, VisualTraj, ImuTraj

# load data
vis_traj = VisualTraj("mono", "./trajs/offline_mandala0_mono.txt")
gt_traj = VisualTraj("gt", "./trajs/offline_mandala0_gt.txt")
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

# measurement noise
R_p = [0.01*0.01 for i in range(3)]
R_q = [0.01 for i in range(3)]
R = np.diag(np.hstack((R_p, R_q)))

# filter main loop
t_start = vis_traj.t[0]
t_end = vis_traj.t[-1]

traj = Trajectory("kf", vis_traj.labels)
for i, t in enumerate(vis_traj.t):
    current_vis = vis_traj.at_index(i)

    # initialisation
    if i == 0:
        x0 = States(p0, v0, q0, bw0, ba0, scale0, p_BC_0, q_BC_0)
        cov0 = np.square(np.diag(stdevs0))

        num_states = x0.size
        num_meas = 7
        num_control = len(u0)

        kf = Filter(num_states, num_meas, num_control)
        kf.dt = 0.
        kf.set_states(x0)
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
            
            traj.append_from_state(ti, kf.states)

            old_ti = ti

    # update
    kf.update(current_vis, R)

    old_t = t

# plot
axes = None
axes = vis_traj.plot(axes)
axes = gt_traj.plot(axes)
axes = traj.plot(axes, min_t=vis_traj.t[0], max_t=vis_traj.t[-1])

plt.legend()
plt.show()