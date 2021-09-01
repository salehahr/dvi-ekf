from Models import Camera, Imu
from Models import RigidSimpleProbe, SymProbe

from Filter import States, Filter

import sys, argparse
import math
import numpy as np
np.set_printoptions(suppress=True, precision=3)

# Simulation params
NUM_KF_RUNS_DEFAULT = 1

# Data generation parameters
""" Number of IMU data between prev. frame up to
    and including the next frame """
NUM_IMU_DEFAULT = 10
""" Number of camera frames to be simulated (None = all) """
NUM_CAM_DEFAULT = None
""" When to cap simulation """
cap_t = None

# Probe model
scope_length        = 50            # [cm]
theta_cam_in_rad    = np.pi / 6     # [rad]

probe = RigidSimpleProbe(scope_length=scope_length,
            theta_cam=theta_cam_in_rad)

# Camera parameters
""" Noise """
STDEV_PC_DEFAULT = [0.2, 0.2, 0.2]                  # [cm]
STDEV_Q_DEG = [0.01, 0.01, 0.7]      # [rad]
STDEV_RC_DEFAULT = np.deg2rad(STDEV_Q_DEG)      # [rad]
""" Scale to convert cam pos to cm in mandala trajectory """
SCALE            = 10

# IMU parameters
""" From MPU 6050 datasheet """
NOISE_SAMPLE_RATE   = 10            # Hz (not output data rate)
GYRO_NOISE          = 0.005 *\
                        math.sqrt(NOISE_SAMPLE_RATE)    # [deg/s]
GRAVITY             = 9.81 * 100                        # [cm/s^2]
ACCEL_NOISE         = 400 * 1e-6 * GRAVITY *\
                        math.sqrt(NOISE_SAMPLE_RATE)    # [cm/s^2]

stdev_om  = [np.deg2rad(GYRO_NOISE)] * 3                # [rad/s]
stdev_acc = [ACCEL_NOISE] * 3                           # [cm/s^2]

# DOFs
""" How much the DOFs are allowed to move in one update.
    (later on in Config.__init__: gets divided by num_interframe_vals
        if not given as an argument to main.py) """
STDEV_DOFS_P_DEFAULT    = 0.25                              # [cm]
STDEV_DOFS_R_deg        = 1                                 # [deg]
STDEV_DOFS_R_DEFAULT    = np.deg2rad(STDEV_DOFS_R_deg)      # [rad]

# Kalman filter parameters
""" Values for initial covariance matrix """
""" Uncertainties of the IMU error states """
stdev_dp            = STDEV_PC_DEFAULT        # [cm]
stdev_dv            = [0.1, 0.1, 0.1]                   # [cm/s]
stdev_dtheta_deg    = [1., 1, 1]                        # [deg]
stdev_dtheta        = np.deg2rad(stdev_dtheta_deg)      # [rad]

""" High initial uncertainties for the error dofs """
imu_rots_deg        = [5, 5, 5]                      # [deg]
imu_rots_in_rad     = np.deg2rad(imu_rots_deg)          # [rad]
stdev_ddofs         = [*imu_rots_in_rad, 10, 10, 10]    # [rad, cm]

""" Uncertainties of the camera error states """
stdev_dp_cam            = STDEV_PC_DEFAULT            # [cm]
stdev_dtheta_cam_deg    = [0.2, 0.2, 0.2]                   # [deg]
stdev_dtheta_cam        = np.deg2rad(stdev_dtheta_cam_deg)  # [rad]

stdevs0 = np.hstack((stdev_dp, stdev_dv, stdev_dtheta, stdev_ddofs, stdev_dp_cam, stdev_dtheta_cam))

""" For tuning process noise and measurement noise matrices """
SCALE_PROCESS_NOISE_DEFAULT = 6e-3

def np_string(arr):
    """ For formatting np arrays when printing. """
    if isinstance(arr, list):
        arr = np.array(arr)

    return np.array2string(arr,
                    precision=4,
                    suppress_small=True)

class Config(object):
    def __init__(self, traj_name=None):
        do_plot_only = traj_name is not None
        if not do_plot_only:
            args = self._parse_arguments()

        # simulation params
        self.num_kf_runs    = args.runs if not do_plot_only \
                                else NUM_KF_RUNS_DEFAULT
        self.mode           = args.m
        do_fast_sim         = bool(args.f) if not do_plot_only else True

        self.max_vals               = 10 if do_fast_sim else args.nc
        self.num_interframe_vals    = 1  if do_fast_sim else args.nb

        """ these get initialised after loading the Camera obj.
            ._gen_sim_params_from_cam() """
        self.min_t          = None
        self.max_t          = None
        self.cap_t          = None
        self.total_data_pts = None

        # probe
        """ Container for probe object containing only the symbolic
            relative kinematics. """
        freeze = [1, 1, 1, 1, 1, 1]
        self.frozen_dofs        = freeze
        self.sym_probe          = SymProbe(probe)
        self.real_joint_dofs    = probe.joint_dofs.copy()
        self.real_imu_dofs      = probe.imu_dofs.copy()
        self.est_imu_dofs_IC    = self.get_dof_IC()

        # noise
        # # process
        self.scale_process_noise    = float(args.kp) if not do_plot_only \
                            else float(SCALE_PROCESS_NOISE_DEFAULT)

        self.stdev_dofs_p   = args.rwp if args.rwp \
                else STDEV_DOFS_P_DEFAULT / self.num_interframe_vals
        self.stdev_dofs_r   = args.rwr if args.rwr \
                else STDEV_DOFS_R_DEFAULT / self.num_interframe_vals

        # # measurement
        self.meas_noise = np.square(np.hstack(
                        (STDEV_PC_DEFAULT, STDEV_RC_DEFAULT)))
        self.q = [*STDEV_PC_DEFAULT, *STDEV_Q_DEG]

        # saving
        self.dof_metric = None
        self.saved_configs = ['scale_process_noise',
                    'meas_noise',
                    'max_vals',
                    'num_interframe_vals',
                    'min_t',
                    'max_t',
                    'total_data_pts',
                    'traj_name',
                    'dof_metric']

        # plot variables
        self.do_plot        = not args.np if not do_plot_only \
                                else do_plot_only
        self.traj_name      = args.traj_name if not traj_name else traj_name
        self.img_filepath_imu = 'img/kf_' + self.img_filename + '_imu.png'
        self.img_filepath_cam = 'img/kf_' + self.img_filename + '_cam.png'
        self.img_filepath_compact   = 'img/kf_' + self.img_filename + \
                                        '_compact.png'
        self.traj_kf_filepath = 'trajs/kf_best_' + self.img_filename + '.txt'
        self.traj_imuref_filepath = 'trajs/imu_ref_' + self.img_filename + '.txt'

    @property
    def img_filename(self):
        return self._gen_img_filename()

    def get_camera(self):
        filepath_cam = f'./trajs/{self.traj_name}.txt'

        if self.max_vals:
            notch = True if self.max_vals > 10 else False
        else:
            notch = True

        cam = Camera(filepath=filepath_cam,
                max_vals=self.max_vals, scale=SCALE, notch=notch)
        self._gen_sim_params_from_cam(cam)

        return cam

    def _gen_sim_params_from_cam(self, camera):
        """ Updates time-related info from camera data. """
        self.max_vals   = camera.max_vals
        self.min_t      = camera.min_t
        self.max_t      = camera.max_t
        self.cap_t      = camera.min_t + cap_t - 1 if cap_t else None
        self.total_data_pts = (self.max_vals - 1) * \
                        self.num_interframe_vals + 1

    def get_imu(self, camera=None, gen_ref=False):
        """ Generates IMU object from interpolated camera data. """
        cam_reference = camera.rotated if camera.has_rotated else camera
        camera_interp = cam_reference.interpolate(self.num_interframe_vals)

        return Imu(probe, camera_interp,
                stdev_acc, stdev_om, gen_ref=gen_ref)

    def get_dof_IC(self):
        """ Generates initial conditions of the IMU dofs,
            which are not the same as the real DOFs.

            DOFs (real) : [ 0.    0.  0.  0.  0. 20.]
            DOFs (IC)   : [ 0.05  0.  0.  3.  3. 23.]
        """
        # dofs0_rot_real = self.real_imu_dofs[:3]
        # dofs0_tr_real  = self.real_imu_dofs[3:]

        # # perturbations
        # delta_ang_rad = np.deg2rad(3)
        # delta_trans_cm = 3

        # # # random perturbations
        # # delta_dof_rot = np.random.normal(loc=0, scale=delta_ang_rad, size=(3,))
        # # delta_dof_tr = np.random.normal(loc=0, scale=delta_trans_cm, size=(3,))

        # # const. perturbations
        # delta_dof_rot = [delta_ang_rad, *dofs0_rot_real[1:]]
        # delta_dof_tr  = [x + delta_trans_cm for x in dofs0_tr_real]

        # dofs0 = [*delta_dof_rot, *delta_dof_tr]

        # dofs0 = [delta_ang_rad, *self.real_imu_dofs[1:]]

        # return dofs0
        # TO CHANGE temporarily set to perfect IC
        return self.real_imu_dofs

    def get_IC(self, imu, camera):
        """ Perfect initial conditions except for DOFs. """
        W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(camera.vec0)

        x0   = States(W_p_BW_0, WW_v_BW_0, R_WB_0,
                        self.est_imu_dofs_IC, camera.p0, camera.q0)
        cov0 = np.square(np.diag(stdevs0))

        return x0, cov0

    def init_filter_objects(self):
        camera      = self.get_camera()
        imu         = self.get_imu(camera, gen_ref=True)
        x0, cov0    = self.get_IC(imu, camera)
        kf          = Filter(self, imu, x0, cov0, self.frozen_dofs)

        return kf, camera, imu

    def _gen_img_filename(self, prop_only=False):
        if prop_only:
            return self.traj_name + '_prop'
        else:
            return self.traj_name + f'_upd_Kp{self.scale_process_noise}'

    def save(self, filename):
        """ Saves come config parameters to text file. """
        configs = '\n'.join([str(self.__dict__[s])
                    for s in self.saved_configs])

        with open(filename, 'w+') as f:
            f.write(configs)

    def read(self, filename):
        """ Reads config parameters from text file. """
        with open(filename, 'r') as f:
            lines = [l.strip() for _, l in enumerate(f)]

        for i, c in enumerate(self.saved_configs):
            if c == 'traj_name':
                self.__dict__[c] = lines[i]
            elif c in ['total_data_pts', 'num_interframe_vals']:
                self.__dict__[c] = int(lines[i])
            else:
                self.__dict__[c] = float(lines[i])

    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description='Run the VI-ESKF.')

        # positional args
        parser.add_argument('traj_name', type=str,
                        help='mandala0_mono, trans_x, rot_x, ...')

        # optional arguments
        parser.add_argument('-f', nargs='?',
                        default=0, const=1, choices=[0, 1], type=int,
                        help='fast sim. (only 10 frames)')
        parser.add_argument('-np', nargs='?',
                        default=0, const=1, choices=[0, 1], type=int,
                        help='no plotting')
        parser.add_argument('-m', metavar='mode',
                        choices=['tune', 'run'], default='run',
                        help=f'tune or run KF')

        parser.add_argument('-nc', default=NUM_CAM_DEFAULT, type=int,
                        help=f'max num of camera values (default: {NUM_CAM_DEFAULT})')
        parser.add_argument('-nb', default=NUM_IMU_DEFAULT, type=int,
                        help=f'num of IMU values b/w frames (default: {NUM_IMU_DEFAULT})')

        parser.add_argument('-kp', default=SCALE_PROCESS_NOISE_DEFAULT,
                        type=float,
                        help=f'scale factor for process noise (default: {SCALE_PROCESS_NOISE_DEFAULT})')

        parser.add_argument('-rwp', type=float,
                        help=f'random walk noise - imu pos - goes into process noise matrix, is generated every propagation step')
        parser.add_argument('-rwr', 
                        type=float,
                        help=f'random walk noise - imu rot - goes into process noise matrix, is generated every propagation step')

        parser.add_argument('-runs', default=NUM_KF_RUNS_DEFAULT,
                        type=int,
                        help=f'num of KF runs (default: {NUM_KF_RUNS_DEFAULT})')

        return parser.parse_args()

    def print_config(self):
        print('Configuration: \n',
                f'\t Trajectory          : {self.traj_name}\n',
                f'\t Mode                : {self.mode}\n\n',
                
                f'\t Num. cam. frames    : {self.max_vals}\n',
                f'\t Num. IMU data       : {self.total_data_pts}\n',
                f'\t(num. IMU b/w frames : {self.num_interframe_vals})\n\n',

                f'\t Frozen DOFs          : {self.frozen_dofs}\n\n',

                f'\t ## Noise values\n',
                f'\t #  P0: Initial process noise\n',
                f'\t std_dp             = {stdev_dp[0]:.1f} \t cm\n',
                f'\t std_dv             = {stdev_dv[0]:.1f} \t cm/s\n',
                f'\t std_dtheta         = {stdev_dtheta_deg[0]:.1f} \t deg\n',
                f'\t std_ddofs_rot      = {imu_rots_deg[0]:.1f} \t deg\n',
                f'\t std_ddofs_trans    = {stdev_ddofs[-1]:.1f} \t cm\n',
                f'\t std_dp_cam         = {stdev_dp_cam[0]:.1f} \t cm\n',
                f'\t std_dtheta_cam     = {stdev_dtheta_cam_deg[0]:.1f} \t deg\n\n',

                f'\t #  Q: IMU measurement noise\n',
                f'\t std_acc    = {ACCEL_NOISE:.1E} cm/s^2\n',
                f'\t std_om     = {GYRO_NOISE:.1E} deg/s\n\n',
                
                f'\t #  Q: IMU dofs random walk noise\n',
                f'\t std_dofs_p = {self.stdev_dofs_p} cm\n',
                f'\t std_dofs_r = {np.rad2deg(self.stdev_dofs_r):.4f} deg\n\n',

                f'\t #  R: Camera measurement noise\n',
                f'\t stdev_pc   = {np_string(self.meas_noise[:3])} cm \n',
                f'\t stdev_qc   = {np_string(np.rad2deg(self.meas_noise[-3:]))} deg\n\n',

                f'\t ## KF tuning\n',
                f'\t k_PROC     = {self.scale_process_noise}\n',
                )
        self.print_dofs()

    def print_dofs(self):
        print(f'DOFs (real) : {np_string(self.real_imu_dofs)}')
        print(f'DOFs (IC)   : {np_string(self.est_imu_dofs_IC)}\n')