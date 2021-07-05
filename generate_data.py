from Filter import VisualTraj, ImuTraj, States
from Models import RigidSimpleProbe, Camera, Imu

import numpy as np
import sympy as sp

# data generation params
max_vals = 50
num_imu_between_frames = 10
# imu_covariance = [0.01, 0.03, 0.01, 0.03, 0.005, 0.005]

# initialise robot
probe_BtoC = RigidSimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

# SLAM data
# filepath_cam = './trajs/offline_mandala0_gt.txt' # stereo
filepath_cam = './trajs/offline_mandala0_mono.txt' # mono
cam = Camera(filepath=filepath_cam, max_vals=max_vals)
cam_interp = cam.interpolate(num_imu_between_frames)
min_t, max_t = cam.t[0], cam.t[-1]

# imu
imu = Imu(probe_BtoC, cam_interp)

# noise matrices
def gen_noise_matrices(Q, Rp, Rq):
    # process noise
    stdev_na = [Q] * 3
    stdev_nw = stdev_na
    stdevs = np.hstack((stdev_na,  stdev_nw))
    Qc = np.square(np.diag(stdevs))

    # measurement noise
    Rp = [Rp] * 3
    Rq = [Rq] * 4
    R = np.diag(np.hstack((Rp, Rq)))

    return Qc, R

# initial states
dofs0 = probe_BtoC.joint_dofs
imu_dofs0 = probe_BtoC.imu_dofs

imu.eval_init(*dofs0)
W_p_BW_0, R_WB_0, _, WW_v_BW_0, _, _ = imu.get_IC()
IC = States(W_p_BW_0, WW_v_BW_0, R_WB_0, imu_dofs0, cam.p0)

# initial covariances
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q))
cov0 = np.square(np.diag(stdevs0))