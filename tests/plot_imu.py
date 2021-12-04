import context
import matplotlib.pyplot as plt
# load data
from generate_data import imu_gt_traj, imu_mono_traj, max_t, min_t

gt_axes = None
gt_axes = imu_gt_traj.noisy.plot(min_t=min_t, max_t=max_t)
gt_axes = imu_gt_traj.plot(gt_axes, min_t=min_t, max_t=max_t)

mono_axes = imu_mono_traj.noisy.plot(min_t=min_t, max_t=max_t)
mono_axes = imu_mono_traj.plot(mono_axes, min_t=min_t, max_t=max_t)

plt.legend()
plt.show()
