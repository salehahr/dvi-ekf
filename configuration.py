from Models import Camera, Imu
from Models import RigidSimpleProbe, SymProbe

from Filter import States

import sys, argparse
import numpy as np
np.set_printoptions(suppress=True, precision=3)

# Data generation parameters
""" Number of IMU data between prev. frame up to
    and including the next frame """
NUM_IMU_DEFAULT = 10
""" Number of camera frames to be simulated """
NUM_CAM_DEFAULT = None
""" When to cap simulation """
cap_t = None

# Probe model
scope_length        = 50            # []
theta_cam_in_rad    = np.pi / 6     # [rad]

probe = RigidSimpleProbe(scope_length=scope_length,
            theta_cam=theta_cam_in_rad)

# Camera parameters
RP_VAL_DEFAULT = 0.1
RQ_VAL_DEFAULT = 0.5
SCALE          = 10

# IMU parameters
stdev_acc = [1e-3] * 3
stdev_om  = [1e-3] * 3

# Kalman filter parameters
""" Values for initial covariance matrix """
stdev_dp        = np.array([0.1, 0.1, 0.1])
stdev_dv        = np.array([0.1, 0.1, 0.1])
stdev_theta     = np.array([0.05, 0.04, 0.025])

imu_rots_in_deg = [30, 30, 30]
imu_rots_in_rad = [r * np.pi / 180 for r in imu_rots_in_deg]
stdev_ddofs     = np.array([*imu_rots_in_rad, 10, 10, 10])      # [rad]

stdev_dp_cam    = np.array([0.1, 0.1, 0.1])
stdev_theta_cam = np.array([0.05, 0.04, 0.025])

stdevs0 = np.hstack((stdev_dp, stdev_dv, stdev_theta, stdev_ddofs, stdev_dp_cam, stdev_theta_cam))

def np_string(arr):
    return np.array2string(arr,
                    precision=2,
                    suppress_small=True)

class Config(object):
    def __init__(self):
        args = self._parse_arguments()

        # probe
        """ Container for probe object containing only the symbolic
            relative kinematics. """
        self.sym_probe       = SymProbe(probe)
        self.real_joint_dofs = probe.joint_dofs

        # noises
        self.Rp_val     = args.Rp
        self.Rq_val     = args.Rq
        self.meas_noise = np.hstack(([self.Rp_val]*3, [self.Rq_val]*4))
        
        # simulation params
        do_fast_sim                 = bool(args.f)
        self.do_prop_only           = args.do_prop_only in ['prop', 'p']

        self.max_vals               = 10 if do_fast_sim else args.nc
        self.num_interframe_vals    = 1  if do_fast_sim else args.nb

        self.min_t          = None
        self.max_t          = None
        self.cap_t          = None
        self.total_data_pts = None

        # plot params
        self.do_plot        = not args.np
        self.traj_name      = args.traj_name
        img_filename        = self._gen_img_filename()
        self.img_filepath_imu = 'img/kf_' + img_filename + '_imu.png'
        self.img_filepath_cam = 'img/kf_' + img_filename + '_cam.png'

    def get_camera(self):
        filepath_cam = f'./trajs/{self.traj_name}.txt'
        cam = Camera(filepath=filepath_cam, max_vals=self.max_vals, scale=SCALE)
        self._gen_sim_params_from_cam(cam)

        return cam

    def _gen_sim_params_from_cam(self, camera):
        self.max_vals   = camera.max_vals
        self.min_t      = camera.min_t
        self.max_t      = camera.max_t
        self.cap_t      = camera.min_t + cap_t - 1 if cap_t else None
        self.total_data_pts = (self.max_vals - 1) * \
                        self.num_interframe_vals + 1

    def get_imu(self, camera=None, gen_ref=False):
        camera_interp = camera.interpolate(self.num_interframe_vals)
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

    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description='Run the VI-ESKF.')

        # positional args
        parser.add_argument('traj_name', type=str,
                        help='mandala0_mono, trans_x, rot_x, ...')
        prop_choices = ['prop', 'p', 'update', 'u', 'pu']
        parser.add_argument('do_prop_only', metavar='prop',
                        choices=prop_choices,
                        help=f'do propagation only or do prop + update;\n{prop_choices}')

        # optional arguments
        parser.add_argument('-f', nargs='?',
                        default=0, const=1, choices=[0, 1], type=int,
                        help='fast sim. (only 10 frames)')
        parser.add_argument('-np', nargs='?',
                        default=0, const=1, choices=[0, 1], type=int,
                        help='no plotting')

        parser.add_argument('-nc', default=NUM_CAM_DEFAULT, type=int,
                        help=f'max num of camera values (default: {NUM_CAM_DEFAULT})')
        parser.add_argument('-nb', default=NUM_IMU_DEFAULT, type=int,
                        help=f'num of IMU values b/w frames (default: {NUM_IMU_DEFAULT})')

        parser.add_argument('-Rp', default=RP_VAL_DEFAULT, type=float,
                        help=f'camera position noise (default: {RP_VAL_DEFAULT})')
        parser.add_argument('-Rq', default=RQ_VAL_DEFAULT,  type=float,
                        help=f'camera rotation noise (default: {RQ_VAL_DEFAULT})')

        return parser.parse_args()

    def print_config(self):
        print('Configuration: \n',
                f'\t Trajectory          : {self.traj_name}\n',
                f'\t Propagate only      : {self.do_prop_only}\n\n',
                
                f'\t Num. cam. frames    : {self.max_vals}\n',
                f'\t Num. IMU data       : {self.total_data_pts}\n',
                f'\t(num. IMU b/w frames : {self.num_interframe_vals})\n\n',

                f'\t ## Noise values\n',
                f'\t #  Initial process noise\n',
                f'\t std_dp             = {stdev_dp}\n',
                f'\t std_dv             = {stdev_dv}\n',
                f'\t std_theta          = {stdev_theta}\n',
                f'\t stdev_ddofs        = {np_string(stdev_ddofs)}\n',
                f'\t stdev_dp_cam       = {stdev_dp_cam}\n',
                f'\t stdev_theta_cam    = {stdev_theta_cam}\n\n',

                f'\t #  IMU measurement noise\n',
                f'\t std_acc    = {stdev_acc}\n',
                f'\t std_om     = {stdev_om}\n\n',
                
                f'\t #  Camera measurement noise\n',
                f'\t cov_pc     = {self.Rp_val}\n',
                f'\t cov_qc     = {self.Rq_val}\n',
                )
