from __future__ import annotations

import casadi
import numpy as np
from spatialmath import SE3

from dvi_ekf.kinematics import equations as eqns
from dvi_ekf.kinematics import symbols as syms
from dvi_ekf.models.trajectory.ImuRefTraj import ImuRefTraj
from dvi_ekf.models.trajectory.ImuTrajectory import ImuTraj

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
    def __init__(self, probe, cam, stdev_na=None, stdev_nom=None, gen_ref=False):
        # from datasheet
        self.stdev_na = stdev_na
        self.stdev_nom = stdev_nom

        # data arrays to be appended to upon evaluating om/acc expression
        self.t = []
        self._om = []
        self._acc = []

        # cam values
        self.cam = cam

        # real forward kinematics (as a function of notch dof)
        self.f_fwkin = casadi.Function(
            "f_fwkin",
            [syms.notchdofs],
            probe.fwkin,
            ["notchdofs"],
            ["p", "R", "v", "om", "acc", "alp"],
        )
        p, R, v, om, acc, alp = probe.fwkin  # om and alp depend on qd6, qdd6

        self.B_p_CB, self.BB_v_CB, self.B_acc_CB = p, v, acc
        self.R_BC, self.B_om_CB, self.B_alp_CB = R, om, alp

        # symbolic expressions
        self.expr = casadi.Function(
            "f_imu_meas",
            [syms.q_cas, syms.qd_cas, syms.qdd_cas, *syms.cam],
            eqns.f_imu_meas(*probe.fwkin),
            ["q", "qd", "qdd", *syms.cam_str],
            ["B_om_BW", "B_acc_BW"],
        )

        # for trajectory reconstruction/plotting
        self.traj = None
        self.ref = ImuRefTraj("imu ref", self) if gen_ref else None
        self._num_imu_between_frames = None

        self._flag_interpolated = False

    @staticmethod
    def create(config, camera, probe, gen_ref=False) -> Imu:
        """Creates synthetic IMU data from interpolated camera data."""
        cam_reference = camera.rotated if camera.rotated else camera
        camera_interp = cam_reference.interpolate(config.interframe_vals)
        return Imu(
            probe,
            camera_interp,
            config.stdev_accel,
            config.stdev_omega,
            gen_ref=gen_ref,
        )

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def reset(self):
        self.t = []
        self._om = []
        self._acc = []

        self.traj.reset()
        self.ref.reset()

    def __repr__(self):
        return f"IMU object containing {self.nvals} values."

    @property
    def nvals(self):
        return len(self.t)

    @property
    def om(self):
        return np.array(self._om).T  # 3xn

    @property
    def acc(self):
        return np.array(self._acc).T  # 3xn

    @property
    def num_imu_between_frames(self):
        self._num_imu_between_frames = self.cam.num_imu_between_frames
        return self._num_imu_between_frames

    @property
    def flag_interpolated(self):
        self._flag_interpolated = self.cam.flag_interpolated
        return self._flag_interpolated

    def clear(self):
        self.t = []
        self._om = []
        self._acc = []

    def eval_expr_single(
        self,
        t,
        q,
        qd,
        qdd,
        W_acc_C,
        R_WC,
        W_om_C,
        W_alp_C,
        append_array=False,
        filepath="",
        overwrite=False,
    ):
        """Evaluates the symbolic expression for 'om' and 'acc' using camera values and DOFs of the probe.

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

        cam = [syms.W_p_CW, R_WC, syms.WW_v_CW, W_om_C, W_acc_C, W_alp_C]

        # fmt: off
        res_om, res_acc = [
            casadi.DM(r).full().reshape(3,)
            for r in self.expr(q, qd, qdd, *cam)
        ]
        # fmt: on

        if append_array:
            self.t.append(t)
            self._om.append(res_om)
            self._acc.append(res_acc)

        if overwrite:
            self.t = [t]
            self._om = [res_om]
            self._acc = [res_acc]

        if filepath:
            with open(filepath, "a+") as f:
                g_str = f"{res_om[0]:.9f} {res_om[1]:.9f} {res_om[2]:.9f}"
                a_str = f"{res_acc[0]:.9f} {res_acc[1]:.9f} {res_acc[2]:.9f} "
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + "\n")

        return res_om, res_acc

    def eval_init(
        self,
        probe_joint_dofs: np.ndarray,
        notch_dofs: np.ndarray,
        overwrite: bool = False,
    ):
        """
        Evaluates initial IMU values using the probe's and DOFs and the notch DOFS.
        :param probe_joint_dofs: initial degrees of freedom of the sensor setup model
        :param notch_dofs: initial degrees of freedom of the notch measurement
        :param overwrite: whether to overwrite current 'om' and 'acc' values
        :return:
        """
        q0, qd0, qdd0 = probe_joint_dofs
        q0[6], qd0[6], qdd0[6] = notch_dofs

        self.eval_expr_single(
            self.cam.t[0],
            q0,
            qd0,
            qdd0,
            self.cam.acc[:, 0],
            self.cam.R[0],
            self.cam.om[:, 0],
            self.cam.alp[:, 0],
            append_array=True,
            overwrite=overwrite,
        )
        self._init_trajectory()

    def _eval_expr(self, append_array=False, filepath=""):
        for n in range(self.cam.max_vals):
            self.eval_expr_single(
                self.cam.t[n],
                self.q,
                self.qd,
                self.qdd,
                self.cam.acc[:, n],
                self.cam.R[n],
                self.cam.om[:, n],
                self.cam.alp[:, n],
                append_array=append_array,
                filepath=filepath,
            )

    def _correct_cam_dims(self, cam_arg):
        return cam_arg.reshape(3, 1) if cam_arg.shape != (3, 1) else cam_arg

    def generate_traj(self, filepath, do_regenerate):
        if do_regenerate:
            self._generate_new_traj(filepath)
        else:
            self._read_from_file(filepath)

    def _generate_new_traj(self, filepath):
        import os
        import time

        t_start = time.process_time()

        print(
            f"Generating IMU data ({self.cam.max_vals} values) and saving to {filepath}..."
        )

        if os.path.exists(filepath):
            os.remove(filepath)

        self.clear()
        self._eval_expr(append_array=True, filepath=filepath)

        self._init_trajectory()

        print(
            f"Time taken to generate data ({self.cam.max_vals} vals): {time.process_time() - t_start:.4f} s."
        )

    def _init_trajectory(self):
        if self.traj is None:
            self.traj = ImuTraj()
        self.traj.t = np.array(self.t)
        self.traj.ax = np.array(self.acc[0, :])
        self.traj.ay = np.array(self.acc[1, :])
        self.traj.az = np.array(self.acc[2, :])
        self.traj.gx = np.array(self.om[0, :])
        self.traj.gy = np.array(self.om[1, :])
        self.traj.gz = np.array(self.om[2, :])

    def reconstruct(self):
        assert self.flag_interpolated == True
        R_WB = [casadi.DM(R_WC @ self.R_BC.T).full() for R_WC in self.cam.R]
        W_p_BW_0, _, _, WW_v_BW_0, _, _ = self.get_IC()
        self.traj.reconstruct(R_WB, W_p_BW_0, WW_v_BW_0)
        return self.traj.reconstructed

    # def get_base(self) -> SE3:
    #     """
    #     [R_WB,    W_p_B]
    #     [0, 0, 0, 1    ]
    #     """
    #     # fmt: off
    #     B_np = np.eye(4)
    #     R_WB_0 = casadi.DM(self.cam.R0 @ self.R_BC.T).full()
    #     B_np[:3, :3] = R_WB_0
    #     B_np[:3, -1] = casadi.DM(self.cam.p0 - R_WB_0 @ self.B_p_CB).full().reshape(3,)
    #     # fmt: on
    #     return SE3(np.array(B_np))

    def ref_vals(self, cam_meas_vec: List[np.ndarray], current_notch: np.ndarray):
        """For troubleshooting.
        Obtains the desired IMU position based on current camera values
        the relative kinematics relations C to B.
        """
        # return W_p_BW, R_WB, WW_v_BW
        current_fwkin = self.f_fwkin(current_notch)
        return [
            casadi.DM(r).full().squeeze()
            for r in eqns.f_imu(*cam_meas_vec, current_notch, *current_fwkin)
        ]

    def write_array_to_file(self, filepath):
        """Writes IMU trajectory, stored in the _om and _acc arrays,
        to a text file.
        """

        with open(filepath, "w+") as f:
            for i, t in enumerate(self.t):
                om_val = self._om[i]
                acc_val = self._acc[i]
                a_str = f"{acc_val[0]:.9f} {acc_val[1]:.9f} {acc_val[2]:.9f} "
                g_str = f"{om_val[0]:.9f} {om_val[1]:.9f} {om_val[2]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + "\n")

    def append_to_file(self, filepath):
        """Appends IMU trajectory to file."""

        with open(filepath, "w+") as f:
            for i, t in enumerate(self.t):
                om_val = self._om[i]
                acc_val = self._acc[i]
                a_str = f"{acc_val[0]:.9f} {acc_val[1]:.9f} {acc_val[2]:.9f} "
                g_str = f"{om_val[0]:.9f} {om_val[1]:.9f} {om_val[2]:.9f}"
                data_str = f"{t:.6f} " + a_str + g_str
                f.write(data_str + "\n")

    def _read_from_file(self, filepath):
        print(f"Reading IMU data from {filepath}...")
        self.traj = ImuTraj(filepath=filepath)
        self._update_from_trajectory()

    def _update_from_trajectory(self):
        self.t = self.traj.t
        self._acc = np.array((self.traj.ax, self.traj.ay, self.traj.az)).T  # nx3
        self._om = np.array((self.traj.gx, self.traj.gy, self.traj.gz)).T  # nx3
