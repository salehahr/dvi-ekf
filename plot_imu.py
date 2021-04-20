import matplotlib.pyplot as plt
from Trajectory import VisualTraj, ImuTraj

max_vals = None
num_imu_between_frames = 30

# # trajectory generation:  stereoGT (ref) and imu
stereoGT_traj = VisualTraj("stereoGT",
        "./trajs/offline_mandala0_gt.txt",
        cap=max_vals)

imu_covariance = [0.01, 0.01, 0.01, 0.07, 0.005, 0.1]
imu_traj = ImuTraj(name='imu gt',
        filepath="./trajs/mandala0_imu.txt",
        vis_data=stereoGT_traj,
        cap=max_vals,
        num_imu_between_frames=num_imu_between_frames,
        covariance=imu_covariance)

# # plot noisy imu and non-noisy imu
min_t = imu_traj.t[0]
max_t = imu_traj.t[-1]

axes = imu_traj.noisy.plot(min_t=min_t, max_t=max_t)
axes = imu_traj.plot(axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()