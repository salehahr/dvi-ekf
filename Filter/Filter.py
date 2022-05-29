from __future__ import annotations

import os
from copy import copy
from typing import TYPE_CHECKING, List, Optional

import casadi
import numpy as np
from tqdm import tqdm

from tools.utils import get_upd_values
from Visuals import FilterPlot

from .context import eqns, syms
from .Quaternion import Quaternion, skew
from .States import ErrorStates
from .Trajectory import FilterTraj

if TYPE_CHECKING:
    from Filter.Simulator import Simulator


class Filter(object):
    """
    Error-State Kalman Filter for fusing camera and IMU measurements
    using a sensor setup model (probe).
    """

    def __init__(self, sim: Simulator):
        assert sim.config.dofs_updated

        # simulation settings
        self.run_id: Optional[int] = None
        self._config = sim.config
        self._dt = 0.0
        self.show_progress = True

        # initial states/covariance
        self._states = sim.x0
        self._P = sim.cov0

        # for casadi: states, inputs
        self._x: List[np.ndarray] = []
        self._u: List[np.ndarray] = []

        # states
        self.num_states = sim.num_states
        self.num_error_states = self.num_states - 2
        self.num_meas = 7
        self.num_noise = 13

        assert self._P.shape == (self.num_error_states, self.num_error_states)

        self._states.frozen_dofs = [bool(fr) for fr in self._config.frozen_dofs]

        # objects
        self._probe = sim.sym_probe
        self.imu = sim.imu

        # imu
        self.imu.eval_init(self._config.gt_joint_dofs, self._states.notch_dofs)
        self.stdev_na = np.array(self.imu.stdev_na)
        self.stdev_nom = np.array(self.imu.stdev_nom)

        # noise: process
        Q = np.eye(self.num_noise)
        Q[0:3, 0:3] = self._dt ** 2 * self.stdev_na ** 2 * np.eye(3)
        Q[3:6, 3:6] = self._dt ** 2 * self.stdev_nom ** 2 * np.eye(3)
        Q[6:13, 6:13] = np.diag(self._config.process_noise_rw_var)
        self.Q = Q

        # noise: measurement
        self.R = np.diag(self._config.meas_noise_var)

        # buffer
        self._om_old = self.imu.om.squeeze()
        self._acc_old = self.imu.acc.squeeze()
        self.R_WB_old = self._states.q.rot

        # static matrices
        self.H = np.zeros([self.num_meas, self.num_error_states])
        self.H[0:6, 18:24] = np.eye(6)
        self.H[6, 15] = 1

        # plot
        self.traj = FilterTraj("kf")
        self.traj.append_propagated_states(self._config.min_t, self._states)

        # metrics
        self.mse = 0
        self.update_mse = 0  # non-cumulative; is overwritten every update stage

    def reset(self, x0, cov0, notch0):
        self._states = copy(x0)
        self._P = np.copy(cov0)
        self._x = []
        self._u = []
        self._dt = 0.0

        # imu / noise
        self.imu.reset()
        self.imu.eval_init(self._config.gt_joint_dofs, notch0)

        self.traj.reset()
        self.traj.append_propagated_states(self._config.min_t, self._states)
        self.mse = 0

    def update_noise_matrices(self):
        Q = np.eye(self.num_noise)
        Q[0:3, 0:3] = self._dt ** 2 * self.stdev_na ** 2 * np.eye(3)
        Q[3:6, 3:6] = self._dt ** 2 * self.stdev_nom ** 2 * np.eye(3)
        Q[6:13, 6:13] = np.diag(self._config.process_noise_rw_var)
        self.Q = Q

        self.R = np.diag(self._config.meas_noise_var)

    @property
    def x(self):
        self._x = self._states.vec
        return self._x

    @property
    def u(self):
        self._u = [self.om_old, self.acc_old]
        return self._u

    @property
    def om_old(self):
        return self._om_old

    @om_old.setter
    def om_old(self, val):
        self._om_old = val.squeeze()

    @property
    def Om_old(self):
        return Quaternion(w=1.0, v=(0.5 * self._dt * self.om_old), do_normalise=True)

    @property
    def acc_old(self):
        return self._acc_old

    @acc_old.setter
    def acc_old(self, val):
        self._acc_old = val.squeeze()

    def run(self, camera, k, run_desc_str):
        """Filter main loop (t>=1) over all camera frames,
        not counting IC.
        arg: k should already be adjusted to start at 1.
        """
        old_t = self._config.min_t
        cam_timestamps = tqdm(
            enumerate(camera.t[1:]),
            total=camera.max_vals,
            initial=1,
            desc=run_desc_str,
            dynamic_ncols=True,
            disable=not self.show_progress,
        )
        self.run_id = k

        for i, t in cam_timestamps:
            i_cam = i + 1  # not counting IC
            self.run_one_epoch(old_t, t, i_cam, camera)
            self.calculate_update_mse(i_cam, camera)

            old_t = t

        # normalise dof_metric
        # self.mse = self.calculate_metric(i_cam, camera)

    def run_one_epoch(self, old_t, t, i_cam, camera):
        """
        Kalman filter run between old camera frame and
        current camera frame.
        1 epoch = (>=1 prediction steps) + (1 update step)
        Calculates the unnormalised DOF MSE.

        i_cam must be already adjusted so that the IC is not counted
        """
        # propagate
        self.propagate_imu(old_t, t)

        # update
        current_cam = camera.at_index(i_cam)  # not counting IC
        ang_notch = camera.get_notch_at(i_cam)[0]
        self.update(t, current_cam, ang_notch)

    def propagate_imu(self, t0, tn):
        """Generates IMU data between old and current camera frame,
        then uses this data to propagate the states as many times
        as there are IMU data between frames.
        """
        cam_queue = self.imu.cam.generate_queue(t0, tn)

        real_probe_dofs = self._config.gt_joint_dofs

        old_ti = t0
        for ii, ti in enumerate(cam_queue.t):
            interp = cam_queue.at_index(ii)
            real_probe_dofs[0][6] = interp.notch
            real_probe_dofs[1][6] = interp.notch_d
            real_probe_dofs[2][6] = interp.notch_dd

            om, acc = self.imu.eval_expr_single(
                ti,
                *real_probe_dofs,
                interp.acc,
                interp.R,
                interp.om,
                interp.alp,
                overwrite=True,
            )

            self.imu.ref.append_value(ti, interp.vec, interp.notch_arr)

            self._dt = ti - old_ti
            self.propagate(ti, om, acc)

            old_ti = ti

    def propagate(self, t, om, acc):
        self._predict_nominal(om, acc)
        self._predict_error()
        self._predict_error_covariance()

        # Buffer
        self.om_old = om
        self.acc_old = acc
        self.R_WB_old = self._states.q.rot

        # for plotting
        self.traj.append_propagated_states(t, self._states)

    def _predict_nominal(self, om, acc):
        est_probe = self._probe.get_est_fwkin(
            self._states.dofs, self._states.notch_dofs
        )

        res = [
            casadi.DM(r).full()
            for r in eqns.f_predict(self._dt, *self.x, *self.u, *est_probe, om, acc)
        ]

        self._states.set(res)

    def _predict_error(self):
        """Calculates jacobian of the error state kinematics w.r.t. error states and w.r.t. noise."""

        Fx = casadi.SX.eye(self.num_error_states)
        Fx[0:3, 3:6] = self._dt * casadi.SX.eye(3)
        Fx[3:6, 6:9] = -self.R_WB_old @ casadi.skew(self.acc_old) * self._dt
        Fx[6:9, 6:9] = self.Om_old.rot.T
        # Fx[9:15, :] # dofs
        Fx[15:17, 16:18] = Fx[15:17, 16:18] + self._dt * casadi.SX.eye(2)

        self.Fx = self._cam_error_jacobian(Fx, syms.err_x)

        # motion model noise jacobian
        Fi = casadi.SX.zeros(self.num_error_states, self.num_noise)
        Fi[3:15, 0:12] = casadi.SX.eye(12)
        Fi[17, 12] = 1
        self.Fi = self._cam_error_jacobian(Fi, syms.n)

        """ returns zero
        self.err_states.set(self.Fx @ self.err_states.vec) """

    def _cam_error_jacobian(self, jac, vars_wrt):
        """Fills the error jacobian (either w.r.t. error state or
        w.r.t. noise) for the camera state entries."""

        err_p_C_next = syms.err_p_C + syms.dt * syms.get_err_pc_dot(self._probe)
        err_theta_C_next = syms.err_theta_C + syms.dt * syms.get_err_theta_c_dot(
            self._probe
        )

        l_in = 0
        r_in = 0
        for x in vars_wrt:
            r_in += x.shape[0]
            jac[18:21, l_in:r_in] = casadi.jacobian(err_p_C_next, x)
            jac[21:, l_in:r_in] = casadi.jacobian(err_theta_C_next, x)
            l_in = r_in

        fun_jac = casadi.Function(
            "f_jac",
            [
                syms.dt,
                syms.dofs,
                syms.notchdofs,
                syms.err_dofs,
                syms.err_notchdofs,
                syms.R_WB,
                *syms.u,
                syms.n_a,
                syms.n_om,
                syms.n_dofs,
                syms.n_notch_acc,
                syms.err_theta,
                syms.err_theta_C,
            ],
            [jac],
            [
                "dt",
                "dofs",
                "notchdofs",
                "err_dofs",
                "err_notchdofs",
                "R_WB",
                *syms.u_str,
                "n_a",
                "n_om",
                "n_dofs",
                "n_notch_acc",
                "err_theta",
                "err_theta_C",
            ],
            ["jac"],
        )
        return casadi.DM(
            fun_jac(
                dt=self._dt,
                dofs=self._states.dofs,
                notchdofs=self._states.notch_dofs,
                R_WB=self.R_WB_old,
                B_om_BW=self.om_old,
                B_acc_BW=self.acc_old,
                n_a=self.stdev_na,
                n_om=self.stdev_nom,
                n_dofs=self._config.process_noise_rw_std[0:6],
                n_notch_acc=self._config.process_noise_rw_std[6],
                # error states have an expectation of zero
                err_dofs=casadi.DM.zeros(
                    6,
                ),
                err_theta=casadi.DM([0.0, 0.0, 0.0]),
                err_theta_C=casadi.DM([0.0, 0.0, 0.0]),
                err_notchdofs=casadi.DM([0.0, 0.0, 0.0]),
            )["jac"]
        ).full()

    def _predict_error_covariance(self):
        """Calculates the predicted covariance.
        Q is calculated from IMU noise and
        from the DOF random walk params.
        """
        self._P = self.Fx @ self._P @ self.Fx.T + self.Fi @ self.Q @ self.Fi.T

    def update(self, t, camera, ang_notch):
        # compute gain
        H = self.H

        S = H @ self._P @ H.T + self.R  # 7x7
        try:
            K = self._P @ H.T @ np.linalg.inv(S)  # 24x7
        except np.linalg.LinAlgError as e:
            print(f"ERROR: {e}!")
            print("Stopping simulation.")
            return None

        # correct virtual SLAM reading to physical SLAM
        notch_quat = Quaternion(val=np.array([0, 0, ang_notch]), euler="xyz")
        cam_rot_corrected = notch_quat * camera.qrot
        # cam_rot_corrected = camera.qrot

        # compute error state
        res_p_cam = camera.pos.reshape((3,)) - self._states.p_cam.reshape((3,))
        err_q = cam_rot_corrected.conjugate * self._states.q_cam
        res_q_cam = err_q.angle * err_q.axis
        res_notch = ang_notch - self._states.notch_dofs[0]
        # res_notch = 0 - self.states.notch_dofs[0]

        res = np.hstack((res_p_cam, res_q_cam, res_notch))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self._states.apply_correction(err)
        I = np.eye(self.num_error_states)
        self._P = (I - K @ H) @ self._P @ (I - K @ H).T + K @ self.R @ K.T

        # reset error states
        G = np.eye(self.num_error_states)  # 24x24
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * err.theta)
        G[-3:, -3:] = np.eye(3) - skew(0.5 * err.theta_c)
        self._P = G @ self._P @ G.T

        # for plotting
        self.traj.append_updated_states(t, self._states)

        return K

    def calculate_update_mse(self, i_cam, camera):
        cam_reference = camera.rotated if camera.rotated else camera
        # current_cam = cam_reference.at_index(i_cam)

        sum_res_cam = 0
        for compc in self.traj.labels_camera[0:6]:
            abs_err = np.array(
                cam_reference.traj.__dict__[compc[:-1]][i_cam]
            ) - get_upd_values(self.traj, compc, -1)
            sum_res_cam += np.square(abs_err)

        sum_res_imu = 0
        for comp in self.traj.labels_imu[3:9]:
            abs_err = get_upd_values(self.traj, comp, -1) - get_upd_values(
                self.imu.ref, comp, -1
            )
            sum_res_imu += np.square(abs_err)

        num_states = 12
        update_mse = (sum_res_cam + sum_res_imu) / num_states

        self.update_mse = update_mse

    def calculate_metric(self, i_cam, camera):
        cam_reference = camera.rotated if camera.rotated else camera

        indices_range = range(0, len(self.traj.t), self._config.interframe_vals)
        upd_indices = [x for x in indices_range]

        sum_res_cam = 0
        for compc in self.traj.labels_camera[0:6]:
            # -1: remove 'c' from string';
            # i_cam + 1: include final value before breaking simulation
            abs_err = np.array(
                cam_reference.traj.__dict__[compc[:-1]][: i_cam + 1]
            ) - get_upd_values(self.traj, compc, upd_indices)
            sum_res_cam += np.sum(np.square(abs_err))

        sum_res_imu = 0
        for comp in self.traj.labels_imu[3:9]:
            abs_err = get_upd_values(
                self.traj, comp, upd_indices[:-1]
            ) - get_upd_values(
                self.imu.ref, comp, upd_indices[:-1]
            )  # no imu ref calculated at end
            sum_res_imu += np.sum(np.square(abs_err))

        num_states = 12
        num_frames = len(upd_indices)
        avg_sum_res = (sum_res_cam + sum_res_imu) / (
            num_states * num_frames
        ) + self.calculate_dof_metric()

        return avg_sum_res

    def calculate_dof_metric(self):
        """at end of run"""
        res = self._states.dofs - self._config.gt_imu_dofs
        return np.dot(res, res) / 6

    def save(self):
        self._config.mse = self.mse
        kf_filepath = os.path.join(
            self._config.traj_path, f"kf_best_{self._config.traj_name}.txt"
        )
        imuref_filepath = kf_filepath.replace("kf_best", "imu_ref")

        self.traj.write_to_file(kf_filepath)
        self.imu.ref.write_to_file(imuref_filepath)

    def plot(self, camera, compact):
        if not self._config.do_plot:
            return

        t_end = self.traj.t[-1]
        plotter = FilterPlot(self, self._config, camera)
        if compact:
            plotter.plot_compact(t_end)
        else:
            plotter.plot(t_end)
