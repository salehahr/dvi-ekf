import context

import matplotlib.pyplot as plt

# load data
from generate_data import stereoGT_traj, mono_traj, imu_gt_traj, imu_mono_traj, min_t, max_t

# # sanity check -- plot interpolated data
# stereoGT_traj.interpolate(num_imu_between_frames)
# axes = stereoGT_traj.plot(min_t=min_t, max_t=max_t)
# axes = stereoGT_traj.interpolated.plot(axes, min_t=min_t, max_t=max_t)

# # sanity check -- plot reconstructed trajectory from imu (coordinates)
recon_gt = imu_gt_traj.reconstruct_vis_traj()
recon_gt_axes = stereoGT_traj.plot(min_t=min_t, max_t=max_t)
recon_gt_axes = recon_gt.plot(recon_gt_axes, min_t=min_t, max_t=max_t)

recon_mono = imu_mono_traj.reconstruct_vis_traj()
recon_mono_axes = mono_traj.plot(min_t=min_t, max_t=max_t)
recon_mono.plot(recon_mono_axes, min_t=min_t, max_t=max_t)

# sanity check -- plot reconstructed trajectory from imu (3D)
recon_gt_axes = stereoGT_traj.plot_3d()
recon_gt_axes = recon_gt.plot_3d(ax=recon_gt_axes)

recon_mono_axes = imu_mono_traj.plot_3d()
recon_mono_axes = recon_gt.plot_3d(ax=recon_mono_axes)

# sanity check -- plot velocities
v_axes = None
v_axes = stereoGT_traj.plot_velocities(v_axes, min_t=min_t, max_t=max_t)
v_axes = imu_gt_traj.plot_velocities(v_axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()