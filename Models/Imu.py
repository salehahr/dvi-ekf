from . import context
from Filter import ImuTraj

import numpy as np
import sympy as sp

from .params import *

""" Notation:
        W_acc_CD    : acceleration of C rel. to D.
                        given in W coordinates
        WB_v_CD     : velocity of C rel. to D, diff. wrt B-CS,
                        given in W coordinates
        R_BW        : rotation which transforms a vector in W to B
    ------
    Angular velocity:
        B_om_BW = R_BW @ W_om_CW + B_om_BC
                = R_BC @ R_WC.T @ W_om_CW - B_om_CB

    Acceleration:
        B_acc_BW = R_BC @ R_WC.T @ W_acc_CW
                        - B_acc_CB
                        - B_alp_BW x B_p_CB
                        - 2 B_om_BW x BB_v_CB
                        - B_om_BW x B_om_BW x B_p_CB
    ------
    Derived terms:
        R_BW        = R_BC @ R_WC.T
        W_p_BW      = W_p_CW - R_WC @ R_BC.T @ B_p_CB

        W_om_BW     = W_om_CW - R_WC @ R_BC.T @ B_om_CB
        WW_v_BW     = WW_v_CW - R_WC @ R_BC.T @ BB_v_CB
                        - W_om_BW x R_WC @ R_BC.T @ B_p_CB

        W_alp_BW    = W_alp_CW - R_WC @ R_BC.T @ B_alp_CB
                        - W_om_BW x R_WC @ R_BC.T @ B_om_CB
    """

class Imu(object):
    def __init__(self, probe_BtoC, cam):
        # data arrays to be appended to upon evaluating om/acc expression
        self.t = []
        self._om = []
        self._acc = []

        # cam values
        self.cam = cam
        # W_p_CW, WW_v_CW, W_acc_CW = cam.p, cam.v, cam.acc
        # R_WC, W_om_CW, W_alp_CW = cam.R, cam.om, cam.alp

        # forward kinematics
        self.probe = probe_BtoC
        self.B_p_CB, self.BB_v_CB, self.B_acc_CB = probe_BtoC.p, probe_BtoC.v, probe_BtoC.acc
        self.R_BC, self.B_om_CB, self.B_alp_CB = probe_BtoC.R, probe_BtoC.om, probe_BtoC.alp

        # derived
        R_BW = self.R_BC @ R_WC_s.T
        B_om_BW = R_BW @ W_om_CW_s - self.B_om_CB
        B_alp_BW = R_BW @ W_alp_CW_s - self.B_alp_CB \
                        - np.cross(B_om_BW, self.B_om_CB, axis=0)

        cross_omBW_pCB = np.cross(B_om_BW, self.B_p_CB, axis=0)
        
        B_acc_BW = R_BW @ W_acc_CW_s \
                        - self.B_acc_CB \
                        - np.cross(B_alp_BW, self.B_p_CB, axis=0) \
                        - 2 * np.cross(B_om_BW, self.BB_v_CB, axis=0) \
                        - np.cross(B_om_BW, cross_omBW_pCB, axis=0)

        # IMU data
        self.om_expr = B_om_BW
        self.acc_expr = B_acc_BW

        # for trajectory reconstruction/plotting
        self.traj = None
        self._num_imu_between_frames = None

        self._flag_interpolated = False

    @property
    def om(self):
        return np.array(self._om).T # 3xn

    @property
    def acc(self):
        return np.array(self._acc).T # 3xn

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
    def flag_interpolated(self):
        self._flag_interpolated = self.cam.flag_interpolated
        return self._flag_interpolated

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def eval_expr(self, append_array=False, filepath=''):
        for n in range(cam.max_vals):
            self.eval_expr_single(self.cam.t[n], self.cam.acc[:,n],
                                self.cam.R[n], self.cam.om[:,n],
                                self.cam.alp[:,n],
                                *self.probe.joint_dofs,
                                append_array=append_array,
                                filepath=filepath)

    def eval_expr_single(self, t, W_acc_C, R_WC, W_om_C, W_alp_C,
        *dofs, append_array=False, filepath=''):
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
        res_om = func_om(W_acc_C, R_WC, W_om_C, W_alp_C, *dofs)

        func_acc = sp.lambdify(params_s, self.acc_expr, modules='sympy')
        res_acc = func_acc(W_acc_C, R_WC, W_om_C, W_alp_C, *dofs)

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

    def reconstruct(self):
        assert(self.flag_interpolated == True)
        R_WB = [R_WC @ self.R_BC.T for R_WC in self.cam.R]
        IC = self._get_IC()
        self.traj.reconstruct(R_WB, *IC)
        return self.traj.reconstructed

    def _get_IC(self):
        """ For trajectory reconstruction.
            Obtains IMU IC based on initial camera values and
            the relative kinematics relations C to B.
        """
        # cam values
        W_p_CW, WW_v_CW, W_acc_CW = self.cam.p0, self.cam.v0, self.cam.acc0
        R_WC, W_om_CW, W_alp_CW = self.cam.R0, self.cam.om0, self.cam.alp0

        R_WB_0 = R_WC @ self.R_BC.T
        W_p_BW_0 = W_p_CW - R_WB_0 @ self.B_p_CB

        W_om_BW_0 = R_WB_0 @ self.om[:,0].reshape(3,1)
        WW_v_BW_0 = WW_v_CW - R_WB_0 @ self.BB_v_CB \
                - np.cross(W_om_BW_0, R_WB_0 @ self.B_p_CB, axis=0)

        W_alp_BW_0 = W_alp_CW - R_WB_0 @ self.B_alp_CB
        W_acc_BW_0 = R_WB_0 @ self.acc[:,0].reshape(3,1)
        
        return W_p_BW_0, W_om_BW_0, WW_v_BW_0, W_alp_BW_0, W_acc_BW_0

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
