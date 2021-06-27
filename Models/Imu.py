from . import context
from Filter import ImuTraj

import numpy as np
import sympy as sp

from .params import *

""" Notation:
        W_acc_B : acceleration of B in world coordinates
        acc_BC  : acceleration of B relative to C,
                    given in world coordinates.
        R_BW    : rotation from W to B
    ------
    Angular velocity:
        W_om_B = W_om_C + om_BC

        B_om_B = R_BW @ (W_om_C + om_BC)

    Acceleration:
        W_acc_B = W_acc_C + acc_BC
                    + W_alp_C x p_BC
                    + 2 W_om_C x v_BC
                    + W_om_C x W_om_C x p_BC

        B_acc_B = R_BW @ W_acc_B
    """

class Imu(object):
    def __init__(self, probe_BtoC, cam):
        self.t = []
        self._om = []
        self._acc = []

        # forward kinematics
        self.probe = probe_BtoC
        R_BC = probe_BtoC.R
        p_BC, v_BC, acc_BC, om_BC, alp_BC = probe_BtoC.get_reversed_kin()
        self.R_BC = R_BC
        self.p_BC = p_BC
        self.v_BC = v_BC
        self.acc_BC = acc_BC
        self.om_BC = om_BC
        self.alp_BC = alp_BC

        self.om_expr = R_BW_s @ (om_C_s + om_BC)
        self.acc_expr = R_BW_s @ (acc_C_s + acc_BC
                    + np.cross(alp_C_s, p_BC, axis=0)
                    + 2 * np.cross(om_C_s, v_BC, axis=0)
                    + np.cross( om_C_s, np.cross(om_C_s, p_BC,
                        axis=0), axis=0 ))

        # for transforming between CS
        self.cam = cam
        self._R_BW = [self.R_BC @ R_WC.T for R_WC in self.cam.R]
        self._R_WB = [R.T for R in self._R_BW]

        # for trajectory reconstruction/plotting
        self.traj = None
        self._num_imu_between_frames = None

        self._W_p0 = [0., 0., 0.]
        self._W_v0 = [0., 0., 0.]
        self._R_WB0 = np.zeros((3, 3))

        self._flag_interpolated = False

    @property
    def om(self):
        return np.array(self._om).T # 3xn

    @property
    def acc(self):
        return np.array(self._acc).T # 3xn

    @property
    def R_BW(self):
        return self._R_BW
    @R_BW.setter
    def R_BW(self, val):
        self._R_BW = val
        self._R_WB = [R.T for R in self._R_BW]
    @property
    def R_WB(self):
        return self._R_WB
    @R_WB.setter
    def R_WB(self, val):
        self._R_WB = val
        self._R_BW = [R.T for R in self._R_WB]

    def init_trajectory(self):
        if self.traj is None:
            self.traj = ImuTraj()
        self.traj.t = self.t
        self.traj.ax = self.acc[0,:]
        self.traj.ay = self.acc[1,:]
        self.traj.az = self.acc[2,:]
        self.traj.gx = self.om[0,:]
        self.traj.gy = self.om[1,:]
        self.traj.gz = self.om[2,:]

    @property
    def num_imu_between_frames(self):
        self._num_imu_between_frames = self.cam.num_imu_between_frames
        return self._num_imu_between_frames

    @property
    def W_p0(self):
        return self._W_p0
    @W_p0.setter
    def W_p0(self, val):
        self._W_p0 = val
        self.traj.W_p0 = val
    @property
    def W_v0(self):
        return self._W_v0
    @W_v0.setter
    def W_v0(self, val):
        self._W_v0 = val
        self.traj.W_v0 = val
    @property
    def R_WB0(self):
        return self._R_WB0
    @R_WB0.setter
    def R_WB0(self, val):
        self._R_WB0 = val
        self.traj.R_WB0 = val

    @property
    def flag_interpolated(self):
        self._flag_interpolated = self.cam.flag_interpolated
        return self._flag_interpolated

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def eval_expr(self, *dofs_array, filepath=''):
        for n in range(cam.max_vals):
            self.eval_expr_single(self.cam.t[n], self.cam.acc[:,n],
                                self.cam.om[:,n], self.cam.alp[:,n],
                                self.R_BW[n],
                                *dofs_array[:,n],
                                append_array=False,
                                filepath=filepath)

    def eval_expr_single(self, t, W_acc_C, W_om_C, W_alp_C, R_BW, *dofs,
            append_array=False, filepath=''):
        """ Evaluates the symbolic expression using camera values and DOFs of the probe.

            Args:
                append_array    : if True, appends the resulting data point (om, acc) to _om and _acc
                filepath        : if given, appends the resulting data point to the file

            Returns:
                res_om  : angular velocity in B coordinates
                res_acc : acceleration in B coordinates
        """
        # dimension check
        W_acc_C = self._correct_cam_dims(W_acc_C)
        W_om_C = self._correct_cam_dims(W_om_C)
        W_alp_C = self._correct_cam_dims(W_alp_C)

        func_om = sp.lambdify(params_s, self.om_expr, modules='sympy')
        res_om = func_om(W_om_C, W_acc_C, W_alp_C, R_BW, *dofs)

        func_acc = sp.lambdify(params_s, self.acc_expr, modules='sympy')
        res_acc = func_acc(W_om_C, W_acc_C, W_alp_C, R_BW, *dofs)

        res_om = np.array(res_om).reshape(3,)
        res_acc = np.array(res_acc).reshape(3,)

        if append_array:
            self.t.append(t)
            self._om.append(res_om)
            self._acc.append(res_acc)

        if filepath:
            with open(filepath, 'a+') as f:
                g_str = f"{res_om[0]:.9f} {res_om[1]:.9f} {res_om[2]:.9f}"
                a_str = f"{res_acc[0]:.9f} {res_acc[1]:.9f} {res_acc[2]:.9f} "
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')

        return res_om, res_acc

    def _correct_cam_dims(self, cam_arg):
        return cam_arg.reshape(3, 1) if cam_arg.shape != (3, 1) else cam_arg

    def reconstruct_traj(self):
        assert(self.flag_interpolated == True)
        self._get_IC()
        self.traj.reconstruct_traj(self.R_BW)
        return self.traj.reconstructed

    def _get_IC(self):
        """ For trajectory reconstruction.
            Obtains IMU IC based on initial camera values and
            the relative kinematics relations C to B.

            W_p_B = W_p_C + W_p_BC
            TODO: check velocity relations,
                    check rel. to which frames are being differentiated!!!
        """
        self.W_p0 = self.cam.p0.reshape(3,1) + self.p_BC
        self.W_v0 = self.cam.v0.reshape(3,1) + self.v_BC \
                    + np.cross(self.cam.om0, self.p_BC, axis=0)
        self.R_WB0 = self.R_BW[0].T

    def write_array_to_file(self, filepath):
        """ Writes IMU trajectory, stored in the _om and _acc arrays,
            to a text file.
        """

        with open(filepath, 'w+') as f:
            for i, t in enumerate(self.t):
                om_val = self._om[i]
                acc_val = self._acc[i]
                a_str = f"{acc_val[0]:.9f} {acc_val[1]:.9f} {acc_val[2]:.9f} "
                g_str = f"{om_val[0]:.9f} {om_val[1]:.9f} {om_val[2]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')

    def append_to_file(self, filepath):
        """ Appends IMU trajectory to file. """

        with open(filepath, 'w+') as f:
            for i, t in enumerate(self.t):
                om_val = self._om[i]
                acc_val = self._acc[i]
                a_str = f"{acc_val[0]:.9f} {acc_val[1]:.9f} {acc_val[2]:.9f} "
                g_str = f"{om_val[0]:.9f} {om_val[1]:.9f} {om_val[2]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + '\n')

    def read_from_file(self, filepath):
        self.traj = ImuTraj(filepath=filepath)
        self._update_from_trajectory()

    def _update_from_trajectory(self):
        self.t = self.traj.t
        self._acc = np.array((self.traj.ax, self.traj.ay, self.traj.az)).T   #nx3
        self._om = np.array((self.traj.gx, self.traj.gy, self.traj.gz)).T    #nx3
