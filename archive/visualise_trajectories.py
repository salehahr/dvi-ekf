"""
Script to visualise camera and IMU trajectories.
"""
import numpy as np
from scipy.integrate import cumtrapz
from spatialmath import UnitQuaternion

from dvi_ekf.models import create_camera
from dvi_ekf.models.Probe import TestProbe
from dvi_ekf.tools import Config
from dvi_ekf.tools.plots import plot_imu
from dvi_ekf.tools.spatial import get_omega_local, interpolate

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
t_interp = np.linspace(camera.min_t, camera.max_t, len(imu_frames_interp))

p = np.array(imu_frames_interp.t)
v = np.gradient(p, t_interp, axis=0)
a = np.gradient(v, t_interp, axis=0)
imu_q_eul = UnitQuaternion(imu_frames_interp).eul()
om_B_local = get_omega_local(imu_frames_interp)

v_recon = cumtrapz(a, t_interp, axis=0, initial=0) + v[0, :]
p_recon = cumtrapz(v_recon, t_interp, axis=0, initial=0) + p[0, :]


dq = UnitQuaternion.Alloc(len(imu_frames_interp))
for i in range(len(imu_frames_interp)):
    try:
        dq[i] = UnitQuaternion.EulerVec(om_B_local[i])
    except TypeError:
        dq[i] = UnitQuaternion()
    except Exception as e:
        raise Exception

q_recon = UnitQuaternion.Alloc(len(imu_frames_interp))
q_recon[0] = UnitQuaternion.Eul(imu_q_eul[0, :])
for i in range(1, len(imu_frames_interp)):
    q_recon[i] = q_recon[i - 1] * dq[i - 1]
eul_recon = q_recon.eul()

plot_imu(t_interp, p, v, a, imu_q_eul, om_B_local, p_recon, v_recon, eul_recon)

# probe.plot(block=True, camera_frames=Wb_F_C, imu_frames=imu_frames_interp, animate=True)
