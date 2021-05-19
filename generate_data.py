from Filter import VisualTraj, ImuTraj, States
import numpy as np

""" Generates visual trajectory data as well as fake IMU data """

max_vals = None
num_imu_between_frames = 300
imu_covariance = [0.01, 0.03, 0.01, 0.03, 0.005, 0.005]

# SLAM data
stereoGT_traj = VisualTraj("stereoGT",
        "./trajs/offline_mandala0_gt.txt",
        cap=max_vals)
mono_traj = VisualTraj("mono",
        "./trajs/offline_mandala0_mono.txt",
        cap=max_vals)

# IMU data
imu_gt_traj = ImuTraj(name='imu gt',
        vis_data=stereoGT_traj,
        cap=max_vals,
        num_imu_between_frames=num_imu_between_frames,
        covariance=imu_covariance)
imu_mono_traj = ImuTraj(name='imu mono',
        vis_data=mono_traj,
        cap=max_vals,
        num_imu_between_frames=num_imu_between_frames,
        covariance=imu_covariance)

# for plotting
min_t = imu_gt_traj.t[0]
max_t = imu_gt_traj.t[-1]

# initial states
p0 = [mono_traj.x[0], mono_traj.y[0], mono_traj.z[0]]
v0 = [stereoGT_traj.vx[0], stereoGT_traj.vy[0], stereoGT_traj.vz[0]]
q0 = [mono_traj.qx[0], mono_traj.qy[0], mono_traj.qz[0],
        mono_traj.qw[0]]

IC = States(p0, v0, q0)

# initial covariances
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q))
cov0 = np.square(np.diag(stdevs0))