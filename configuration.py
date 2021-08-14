from Models import Camera, Imu
from Models import RigidSimpleProbe, SymProbe

from Filter import States

import sys
import numpy as np

# Data generation parameters
""" Number of camera frames to be simulated """
max_vals = None

""" When to cap simulation """
cap_t = None

""" Number of IMU data between prev. frame up to
    and including the next frame """
num_imu_between_frames = 10

# Probe model
scope_length        = 50            # []
theta_cam_in_rad    = np.pi / 6     # [rad]

probe = RigidSimpleProbe(scope_length=scope_length,
            theta_cam=theta_cam_in_rad)
            
# Camera parameters
RP_VAL_DEFAULT = 0.1
RQ_VAL_DEFAULT = 0.5

# IMU parameters
stdev_acc = [1e-3] * 3
stdev_om  = [1e-3] * 3

# Kalman filter parameters
""" Values for initial covariance matrix """
stdev_p = [0.1, 0.1, 0.1]
stdev_v = [0.1, 0.1, 0.1]
stdev_q = [0.05, 0.04, 0.025]
stdev_dofs = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
stdev_p_cam = [0.1, 0.1, 0.1]
stdev_q_cam = [0.05, 0.04, 0.025]

stdevs0 = np.hstack((stdev_p, stdev_v, stdev_q, stdev_dofs, stdev_p_cam, stdev_q_cam))

class Config(object):
    def __init__(self, main_filename):
        # params from user arguments
        self.traj_name      = ''
        self.do_prop_only   = False
        self.Rp_val         = RP_VAL_DEFAULT
        self.Rq_val         = RQ_VAL_DEFAULT
        self._parse_arguments(main_filename)
        
        self.max_vals               = max_vals
        self.num_imu_between_frames = num_imu_between_frames

        # probe
        """ Container for probe object containing only the symbolic
            relative kinematics. """
        self.sym_probe       = SymProbe(probe)
        self.real_joint_dofs = probe.joint_dofs
        
        # noises
        self.meas_noise = np.hstack(([self.Rp_val]*3, [self.Rq_val]*4))
        
        # simulation params
        self.min_t          = None
        self.max_t          = None
        self.cap_t          = None
        self.total_data_pts = None

        # plot params
        img_filename = self._gen_img_filename()
        self.img_filepath_imu = 'img/kf_' + img_filename + '_imu.png'
        self.img_filepath_cam = 'img/kf_' + img_filename + '_cam.png'

    def get_camera(self):
        filepath_cam = f'./trajs/{self.traj_name}.txt'
        cam = Camera(filepath=filepath_cam, max_vals=self.max_vals)
        self._gen_sim_params_from_cam(cam)

        return cam

    def _gen_sim_params_from_cam(self, camera):
        self.max_vals   = camera.max_vals
        self.min_t      = camera.min_t
        self.max_t      = camera.max_t
        self.cap_t      = camera.min_t + cap_t - 1 if cap_t else None
        self.total_data_pts = (self.max_vals - 1) * \
                        self.num_imu_between_frames + 1

    def get_imu(self, camera=None, gen_ref=False):
        camera_interp = camera.interpolate(self.num_imu_between_frames)
        return Imu(probe, camera_interp, stdev_acc, stdev_om, gen_ref=gen_ref)

    def get_IC(self, imu, camera):
        W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(camera.vec0)
        dofs0 = probe.imu_dofs.copy()

        IC   = States(W_p_BW_0, WW_v_BW_0, R_WB_0,
                        dofs0, camera.p0, camera.q0)
        cov0 = np.square(np.diag(stdevs0))

        return IC, cov0

    def _gen_img_filename(self):
        if self.do_prop_only:
            return self.traj_name + '_prop'
        else:
            return self.traj_name + f'_upd_Rp{self.Rp_val}_Rq{self.Rq_val}'

    def _parse_arguments(self, main_filename):
        def print_usage():
            print(f"Usage: {main_filename}",
                "<traj_name> <prop> <Rpval> <Rqval>\n\n",
                
                    "\t<traj_name> : mandala0_mono, trans_x, rot_x, ...\n",
                    "\t<prop>      : prop / update\n\n",
                
                "Optional:\n",
                    "\t <Rpval>    : default is 1e2\n",
                    "\t <Rqval>    : default is 0.5")
            sys.exit()

        try:
            self.traj_name = sys.argv[1]
            self.do_prop_only = False or (sys.argv[2] == 'prop')
        except IndexError:
            print_usage()

        # optional args
        if len(sys.argv) > 3:
            self._parse_meas_noise_from_args()

    def _parse_meas_noise_from_args(self):
        self.Rp_val = float(sys.argv[3])
        self.Rq_val = float(sys.argv[4])

    def print_config(self):
        print('Configuration: \n',
                f'\t Trajectory          : {self.traj_name}\n',
                f'\t Propagate only      : {self.do_prop_only}\n\n',
                
                f'\t Num. cam. frames    : {self.max_vals}\n',
                f'\t Num. IMU data       : {self.total_data_pts}\n',
                f'\t(num. IMU b/w frames : {self.num_imu_between_frames})\n\n',

                f'\t ## Noise values\n',
                f'\t #  IMU measurement noise\n',
                f'\t std_acc    = {stdev_acc}\n',
                f'\t std_om     = {stdev_om}\n\n',
                
                f'\t #  Camera measurement noise\n',
                f'\t cov_p      = {self.Rp_val}\n',
                f'\t cov_q      = {self.Rq_val}\n',
                )
        print(f'----------------------------')
