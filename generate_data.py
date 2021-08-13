from Filter import VisualTraj, ImuTraj, States, ImuRefTraj
from Models import RigidSimpleProbe, SymProbe, Camera, Imu

import numpy as np
import sympy as sp

# data generation params
max_vals = 50
num_imu_between_frames = 10
# imu_covariance = [0.01, 0.03, 0.01, 0.03, 0.005, 0.005]

# imu params
stdev_na, stdev_nom = [1e-3]*3, [1e-3]*3

# initialise robot
probe = RigidSimpleProbe(scope_length=50, theta_cam=sp.pi/6)
sym_probe = SymProbe(probe)

def get_cam(traj_name):
    filepath_cam = f'./trajs/{traj_name}.txt' # simple traj smooth
    cam = Camera(filepath=filepath_cam, max_vals=max_vals)
    cam_interp = cam.interpolate(num_imu_between_frames)
    return cam, cam_interp

def get_imu(cam_interp):
    imu = Imu(probe, cam_interp, stdev_na, stdev_nom)
    imu_ref = ImuRefTraj("imu ref", imu)
    return imu, imu_ref

def get_IC(imu, cam, probe):
    W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(cam.vec0)
    dofs0 = probe.imu_dofs.copy()
    return States(W_p_BW_0, WW_v_BW_0, R_WB_0, dofs0, cam.p0, cam.q0)

def get_data(traj_name):
    cam, cam_interp = get_cam(traj_name)
    cam0 = cam.vec_at(0)
    min_t, max_t = cam.t[0], cam.t[-1]

    imu, imu_ref = get_imu(cam_interp)
    imu_ref.append_value(min_t, cam.vec0)

    IC = get_IC(imu, cam, probe)

    return cam, cam_interp, imu, imu_ref, IC, min_t, max_t

# initial covariances
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]
stdev_dofs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
stdev_p_cam = [0.1, 0.1, 0.1]
stdev_q_cam = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q, stdev_dofs, stdev_p_cam, stdev_q_cam))
cov0 = np.square(np.diag(stdevs0))