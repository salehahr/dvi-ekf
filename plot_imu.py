import matplotlib.pyplot as plt
from Trajectory import VisualTraj, ImuTraj

stereoGT_traj = VisualTraj("stereoGT", "./trajs/offline_mandala0_gt.txt")
imu_covariance = [0.004, 0.003, 0.002, 0.07, 0.005, 0.07]

imu_traj = ImuTraj(name='imu gt', filepath="./trajs/mandala0_imu.txt",
        vis_data=stereoGT_traj, num_imu_between_frames=100,
        covariance=imu_covariance)

min_t = imu_traj.t[0]
max_t = imu_traj.t[-1]

axes = imu_traj.noisy.plot(min_t=min_t, max_t=max_t)
axes = imu_traj.plot(axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()