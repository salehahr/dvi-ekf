"""
Script to visualise camera and IMU trajectories.
"""
import numpy as np

from Models import create_camera
from Models.Probe import TestProbe
from tools import Config
from tools.spatial import get_omega_local, interpolate

# config
config = Config("config.yaml")

# models
camera = create_camera(config)
probe = TestProbe(scope_length=config.model.length, theta_cam=config.model.angle)
Wb_F_Wc = probe.T

# frames in probe base coordinates
Wb_F_C = Wb_F_Wc * camera.frames
Wb_F_B = Wb_F_C * probe.T.inv()

n_interframe_vals = 5
imu_frames_interp = interpolate(Wb_F_B, n_interframe_vals)
imu_pos = np.array(imu_frames_interp.t)
t_interp = np.linspace(camera.min_t, camera.max_t, len(imu_frames_interp))
v = np.gradient(imu_pos, t_interp, axis=0)
a = np.gradient(v, t_interp, axis=0)
om_B_local = get_omega_local(imu_frames_interp)

# probe.plot(block=True, camera_frames=Wb_F_C, imu_frames=imu_frames_interp, animate=True)
