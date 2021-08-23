from Models import Camera, Imu
from Models import RigidSimpleProbe, SymProbe

from Filter import States

import sys, argparse
import math
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
scope_length        = 50            # [cm]
theta_cam_in_rad    = np.pi / 6     # [rad]

probe = RigidSimpleProbe(scope_length=scope_length,
            theta_cam=theta_cam_in_rad)

# Camera parameters
STDEV_PC_DEFAULT = 0.3              # [cm]
STDEV_RC_DEFAULT = np.deg2rad(5)    # [rad]
SCALE            = 10               # convert cam pos to cm

# IMU parameters
NOISE_SAMPLE_RATE   = 10            # Hz (not output data rate)
GYRO_NOISE          = 0.005 *\
                        math.sqrt(NOISE_SAMPLE_RATE)    # [deg/s]
GRAVITY             = 9.81 * 100                        # [cm/s^2]
ACCEL_NOISE         = 400 * 1e-6 * GRAVITY *\
                        math.sqrt(NOISE_SAMPLE_RATE)    # [cm/s^2]

stdev_om  = [np.deg2rad(GYRO_NOISE)] * 3                # [rad/s]
stdev_acc = [ACCEL_NOISE] * 3                           # [cm/s^2]

# DOFs
STDEV_DOFS_P        = 1                                 # [cm]
STDEV_DOFS_R_deg    = 3                                 # [deg]
STDEV_DOFS_R        = np.deg2rad(STDEV_DOFS_R_deg)      # [rad]

# Kalman filter parameters
""" Values for initial covariance matrix """
stdev_dp            = [STDEV_PC_DEFAULT * 3] * 3      # [cm]
stdev_dv            = [0.1, 0.1, 0.1]                   # [cm/s]
stdev_dtheta_deg    = [1., 1, 1]                         # [deg]
stdev_dtheta        = np.deg2rad(stdev_dtheta_deg)      # [rad]

imu_rots_deg        = [30, 30, 30]                      # [deg]
imu_rots_in_rad     = np.deg2rad(imu_rots_deg)          # [rad]
stdev_ddofs         = [*imu_rots_in_rad, 10, 10, 10]    # [rad, cm]

stdev_dp_cam            = [STDEV_PC_DEFAULT * 3] * 3  # [cm]
stdev_dtheta_cam_deg    = [0.2, 0.2, 0.2]               # [deg]
stdev_dtheta_cam        = np.deg2rad(stdev_dtheta_cam_deg) # [rad]

stdevs0 = np.hstack((stdev_dp, stdev_dv, stdev_dtheta, stdev_ddofs, stdev_dp_cam, stdev_dtheta_cam))

""" For tuning process noise and measurement noise matrices """
SCALE_PROCESS_NOISE = 1
SCALE_MEASUREMENT_NOISE = 1

def np_string(arr):
    if isinstance(arr, list):
        arr = np.array(arr)

    return np.array2string(arr,
                    precision=4,
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
        # # process
        self.scale_process_noise    = args.kp
        self.stdev_dofs_p           = STDEV_DOFS_P
        self.stdev_dofs_r           = STDEV_DOFS_R

        # # measurement
        self.scale_meas_noise       = args.km
        self.Rpc_val                = args.Rp
        self.Rqc_val                = args.Rq
        self.meas_noise             = np.hstack(([self.Rpc_val**2]*3,
                                        [self.Rqc_val**2]*3))
        
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
            return self.traj_name + f'_upd_Kp{self.scale_process_noise}_Km{self.scale_meas_noise:.3f}'

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

        parser.add_argument('-Rp', default=STDEV_PC_DEFAULT, type=float,
                        help=f'camera position noise (default: {STDEV_PC_DEFAULT:.3f})')
        parser.add_argument('-Rq', default=STDEV_RC_DEFAULT,  type=float,
                        help=f'camera rotation noise (default: {STDEV_RC_DEFAULT:.3f})')

        parser.add_argument('-kp', default=SCALE_PROCESS_NOISE, type=float,
                        help=f'scale factor for process noise (default: {SCALE_PROCESS_NOISE})')
        parser.add_argument('-km', default=SCALE_MEASUREMENT_NOISE,
                        type=float,
                        help=f'scale factor for measurement noise (default: {SCALE_MEASUREMENT_NOISE})')

        return parser.parse_args()

    def print_config(self):
        print('Configuration: \n',
                f'\t Trajectory          : {self.traj_name}\n',
                f'\t Propagate only      : {self.do_prop_only}\n\n',
                
                f'\t Num. cam. frames    : {self.max_vals}\n',
                f'\t Num. IMU data       : {self.total_data_pts}\n',
                f'\t(num. IMU b/w frames : {self.num_interframe_vals})\n\n',

                f'\t ## Noise values\n',
                f'\t #  P0: Initial process noise\n',
                f'\t std_dp             = {np_string(stdev_dp)} \t cm\n',
                f'\t std_dv             = {np_string(stdev_dv)} \t cm/s\n',
                f'\t std_theta          = {np_string(stdev_dtheta_deg)} \t deg\n',
                f'\t std_ddofs_rot      = {np_string(imu_rots_deg)} \t deg\n',
                f'\t std_ddofs_trans    = {np_string(stdev_ddofs[-3:])} \t cm\n',
                f'\t std_dp_cam         = {np_string(stdev_dp_cam)} \t cm\n',
                f'\t std_dtheta_cam     = {np_string(stdev_dtheta_cam_deg)} \t deg\n\n',

                f'\t #  Q: IMU measurement noise\n',
                f'\t std_acc    = {np_string(stdev_acc)} cm/s^2\n',
                f'\t std_om     = {np_string(stdev_om)} rad/s\n\n',
                
                f'\t #  Q: IMU dofs random walk noise\n',
                f'\t std_dofs_p = {self.stdev_dofs_p} cm\n',
                f'\t std_dofs_r = {STDEV_DOFS_R_deg} deg\n\n',

                f'\t #  R: Camera measurement noise\n',
                f'\t stdev_pc   = {self.Rpc_val:.3f} cm \n',
                f'\t stdev_qc   = {self.Rqc_val:.3f} rad\n\n',

                f'\t ## KF tuning\n',
                f'\t k_PROC     = {self.scale_process_noise}\n',
                f'\t k_MEAS     = {self.scale_meas_noise}\n',
                )
