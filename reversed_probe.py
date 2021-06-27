from Models import ReversedProbe
import numpy as np

# trajectory: rotate scope in a semi circle and reverse
max_rot = 1.3*np.pi
q_notch = np.hstack([np.zeros((5,)),
                np.linspace(0, max_rot, num=25),
                np.ones((10,)) * max_rot,
                np.linspace(max_rot, 0, num=25)])

q_cam_imu = np.zeros(q_notch.shape)

# q_slam = np.zeros(q_notch.shape) # unconstrained
q_slam = -1 * q_notch # constrain SLAM coords to be 'upright' (like the real camera)

q = np.array([q_slam, q_notch, q_cam_imu, q_cam_imu]).T

# probe
probe = ReversedProbe(scope_length=0.6, theta_cam=30*np.pi/180)
gif_filepath = None # './img/probe_rev_con.gif'
probe.plot(q, dt=0.05, movie=gif_filepath)
