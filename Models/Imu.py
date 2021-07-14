from . import context
from Filter import ImuTraj

import numpy as np
import sympy as sp
import casadi

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
        R_BW = self.R_BC @ R_WC_cas.T
        B_om_BW = R_BW @ W_om_CW_cas - self.B_om_CB
        B_alp_BW = R_BW @ W_alp_CW_cas - self.B_alp_CB \
                        - casadi.cross(B_om_BW, self.B_om_CB)

        cross_omBW_pCB = casadi.cross(B_om_BW, self.B_p_CB)
        
        B_acc_BW = R_BW @ W_acc_CW_cas \
                        - self.B_acc_CB \
                        - casadi.cross(B_alp_BW, self.B_p_CB) \
                        - 2 * casadi.cross(B_om_BW, self.BB_v_CB) \
                        - casadi.cross(B_om_BW, cross_omBW_pCB)

        # IMU data
        self.om_expr = casadi.Function('f_B_om_BW',
                [q_cas, qd_cas, R_WC_cas, W_om_CW_cas],
                [B_om_BW],
                    ['q', 'qd', 'R_WC', 'om_C'],
                    ['B_om_BW'])
        self.acc_expr = casadi.Function('f_B_acc_BW',
                [q_cas, qd_cas, qdd_cas,
                    R_WC_cas, W_om_CW_cas, W_acc_CW_cas, W_alp_CW_cas],
                [B_acc_BW],
                    ['q', 'qd', 'qdd', 'R_WC', 'om_C', 'acc_C', 'alp_C'],
                    ['B_acc_BW'])

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

    def eval_expr_single(self, t, q, qd, qdd,
        W_acc_C, R_WC, W_om_C, W_alp_C,
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

        res_om = casadi.DM(self.om_expr(q, qd, R_WC, W_om_C)).full().reshape(3,)
        res_acc = casadi.DM(self.acc_expr(q, qd, qdd,
                        R_WC, W_om_C, W_acc_C, W_alp_C)).full().reshape(3,)

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

    def eval_init(self, q, qd, qdd):
        self.eval_expr_single(self.cam.t[0], q, qd, qdd,
                                self.cam.acc[:,0],
                                self.cam.R[0], self.cam.om[:,0],
                                self.cam.alp[:,0],
                                append_array=True)
        self._init_trajectory()

    def _eval_expr(self, append_array=False, filepath=''):
        for n in range(self.cam.max_vals):
            self.eval_expr_single(self.cam.t[n], self.probe.q_cas,
                self.probe.qd_cas, self.probe.qdd_cas,
                self.cam.acc[:,n], self.cam.R[n],
                self.cam.om[:,n], self.cam.alp[:,n],
                append_array=append_array, filepath=filepath)

    def _correct_cam_dims(self, cam_arg):
        return cam_arg.reshape(3, 1) if cam_arg.shape != (3, 1) else cam_arg

    def generate_traj(self, filepath, do_regenerate):
        if do_regenerate:
            self._generate_new_traj(filepath)
        else:
            self._read_from_file(filepath)

    def _generate_new_traj(self, filepath):
        import time, os

        t_start = time.process_time()

        print(f"Generating IMU data ({self.cam.max_vals} values) and saving to {filepath}...")

        if os.path.exists(filepath):
            os.remove(filepath)

        self._eval_expr(append_array=True, filepath=filepath)

        self._init_trajectory()

        print(f"Time taken to generate data ({self.cam.max_vals} vals): {time.process_time() - t_start:.4f} s.")

    def _init_trajectory(self):
        if self.traj is None:
            self.traj = ImuTraj()
        self.traj.t = np.array(self.t)
        self.traj.ax = np.array(self.acc[0,:])
        self.traj.ay = np.array(self.acc[1,:])
        self.traj.az = np.array(self.acc[2,:])
        self.traj.gx = np.array(self.om[0,:])
        self.traj.gy = np.array(self.om[1,:])
        self.traj.gz = np.array(self.om[2,:])

    def reconstruct(self):
        assert(self.flag_interpolated == True)
        R_WB = [casadi.DM(R_WC @ self.R_BC.T).full() for R_WC in self.cam.R]
        W_p_BW_0, _, _, WW_v_BW_0, _, _ = self.get_IC()
        self.traj.reconstruct(R_WB, W_p_BW_0, WW_v_BW_0)
        return self.traj.reconstructed

    def get_IC(self):
        """ For trajectory reconstruction.
            Obtains IMU IC based on initial camera values and
            the relative kinematics relations C to B.
        """
        # cam values
        W_p_CW, WW_v_CW, W_acc_CW = self.cam.p0, self.cam.v0, self.cam.acc0
        R_WC, W_om_CW, W_alp_CW = self.cam.R0, self.cam.om0, self.cam.alp0

        R_WB_0 = casadi.DM(R_WC @ self.R_BC.T).full()
        W_p_BW_0 = casadi.DM(W_p_CW - R_WB_0 @ self.B_p_CB).full()

        W_om_BW_0 = casadi.DM(R_WB_0 @ self.om[:,0].reshape(3,1)).full()
        WW_v_BW_0 = casadi.DM(WW_v_CW - R_WB_0 @ self.BB_v_CB \
                - casadi.cross(W_om_BW_0, R_WB_0 @ self.B_p_CB)).full()

        W_alp_BW_0 = casadi.DM(W_alp_CW - R_WB_0 @ self.B_alp_CB).full()
        W_acc_BW_0 = casadi.DM(R_WB_0 @ self.acc[:,0].reshape(3,1)).full()

        return W_p_BW_0, R_WB_0, W_om_BW_0, WW_v_BW_0, W_alp_BW_0, W_acc_BW_0

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

    def _read_from_file(self, filepath):
        print(f"Reading IMU data from {filepath}...")
        self.traj = ImuTraj(filepath=filepath)
        self._update_from_trajectory()

    def _update_from_trajectory(self):
        self.t = self.traj.t
        self._acc = np.array((self.traj.ax, self.traj.ay, self.traj.az)).T   #nx3
        self._om = np.array((self.traj.gx, self.traj.gy, self.traj.gz)).T    #nx3
