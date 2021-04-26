from Filter import VisualTraj, ImuTraj

""" Generates visual trajectory data as well as fake IMU data """

max_vals = None
num_imu_between_frames = 10
imu_covariance = [0.01, 0.01, 0.01, 0.07, 0.005, 0.1]

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