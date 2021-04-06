import matplotlib.pyplot as plt
from Trajectory import VisualTraj, ImuTraj

stereoGT_traj_filepath = "./trajs/offline_mandala0_gt.txt"
stereoGT_traj = VisualTraj("stereoGT", stereoGT_traj_filepath)

imu_data = {
            'from_gt': "./trajs/mandala0_imu.txt",
            'noisy': "./trajs/mandala0_imu_noisy.txt",
            }

imu_traj = ImuTraj(filepath=imu_data['from_gt'],
        vis_data=stereoGT_traj, num_imu_between_frames=100)

axes = None
min_t = imu_traj.t[0]
max_t = imu_traj.t[-1]
for name, filepath in imu_data.items():
    traj = ImuTraj(name=name, filepath=filepath)
    axes = traj.plot(axes, min_t, max_t)

plt.legend()
plt.show()