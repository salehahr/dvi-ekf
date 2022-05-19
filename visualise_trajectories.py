"""
Script to visualise camera and IMU trajectories.
"""

from Models import create_camera
from Models.Probe import TestProbe
from tools import Config

# config
config = Config("config.yaml")

# models
camera = create_camera(config)
probe = TestProbe(scope_length=config.model.length, theta_cam=config.model.angle)
Wb_F_Wc = probe.T

# frames in probe base coordinates
Wb_F_C = Wb_F_Wc * camera.frames
Wb_F_B = Wb_F_C * probe.T.inv()

probe.plot(block=True, camera_frames=Wb_F_C, imu_frames=Wb_F_B, animate=True)
