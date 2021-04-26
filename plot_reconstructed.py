import matplotlib.pyplot as plt
from Trajectory import VisualTraj, ImuTraj

max_vals = None
num_imu_between_frames = 10

# trajectory generation:  stereoGT (ref) and imu
stereoGT_traj = VisualTraj("stereoGT",
        "./trajs/offline_mandala0_gt.txt",
        cap=max_vals)
mono_traj = VisualTraj("mono",
        "./trajs/offline_mandala0_mono.txt",
        cap=max_vals)

imu_covariance = [0.01, 0.01, 0.01, 0.07, 0.005, 0.1]
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

# plots
min_t = imu_gt_traj.t[0]
max_t = imu_gt_traj.t[-1]

# # sanity check -- plot interpolated data
# stereoGT_traj.interpolate(num_imu_between_frames)
# axes = stereoGT_traj.plot(min_t=min_t, max_t=max_t)
# axes = stereoGT_traj.interpolated.plot(axes, min_t=min_t, max_t=max_t)

# sanity check -- plot reconstructed trajectory from imu
recon_gt = imu_gt_traj.reconstruct_vis_traj()
recon_gt_axes = stereoGT_traj.plot(min_t=min_t, max_t=max_t)
recon_gt.plot(recon_gt_axes, min_t=min_t, max_t=max_t)

recon_mono = imu_mono_traj.reconstruct_vis_traj()
recon_mono_axes = mono_traj.plot(min_t=min_t, max_t=max_t)
recon_mono.plot(recon_mono_axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()