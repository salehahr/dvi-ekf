from Filter import VisualTraj, ImuTraj, States, ImuRefTraj
from Models import RigidSimpleProbe, SymProbe, Camera, Imu

import numpy as np
import sympy as sp

# data generation params
max_vals = 50
num_imu_between_frames = 10
# imu_covariance = [0.01, 0.03, 0.01, 0.03, 0.005, 0.005]

# initialise robot
probe = RigidSimpleProbe(scope_length=50, theta_cam=sp.pi/6)
sym_probe = SymProbe(probe)

# SLAM data
# filepath_cam = './trajs/offline_mandala0_gt.txt' # stereo

# traj_name = 'mandala0_mono'
# traj_name = 'from_prop'
# traj_name = 'trans_z'
traj_name = 'rot_x'

filepath_cam = f'./trajs/{traj_name}.txt' # simple traj smooth

cam = Camera(filepath=filepath_cam, max_vals=max_vals)
cam_interp = cam.interpolate(num_imu_between_frames)
min_t, max_t = cam.t[0], cam.t[-1]
cam0 = cam.vec_at(0)

# imu
stdev_na, stdev_nom = [1e-3]*3, [1e-3]*3 # supposedly from IMU datasheet
imu = Imu(probe, cam_interp, stdev_na, stdev_nom)
imu_ref = ImuRefTraj("imu ref", imu)
imu_ref.append_value(min_t, cam0)

# initial states
W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(cam0)
IC = States(W_p_BW_0, WW_v_BW_0, R_WB_0, probe.imu_dofs, cam.p0, cam.q0)

# initial covariances
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]
stdev_dofs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
stdev_p_cam = [0.1, 0.1, 0.1]
stdev_q_cam = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q, stdev_dofs, stdev_p_cam, stdev_q_cam))
cov0 = np.square(np.diag(stdevs0))