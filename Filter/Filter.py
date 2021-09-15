from copy import copy, deepcopy
from math import factorial
import numpy as np
from tqdm import tqdm

from .Quaternion import Quaternion, skew
from .Trajectory import FilterTraj
from .States import States, ErrorStates

import casadi
from .context import syms, eqns
from Visuals import FilterPlot

UPDATE_MSE_THRESHOLD = 1
SAVE_STATES_FRAME = 10

class Filter(object):
    def __init__(self, sim):
        # simulation
        self.sim = sim
        self.show_progress = True
        self._run_id = None
        self.dt = 0.

        # initial states/covariance
        self.states = deepcopy(sim.x0)
        self.P = np.copy(sim.cov0)

        # states
        self.num_states = sim.x0.size
        self.num_error_states = self.num_states - 2
        self.num_meas = 7
        self.num_noise = 13

        assert(self.P.shape == (self.num_error_states, self.num_error_states))

        self.states.frozen_dofs = [bool(fr) for fr in self.config.frozen_dofs]
        self._x = []
        self._u = []

        # objects
        self.probe = self.config.sym_probe
        self.camera = sim.camera
        self.imu = sim.imu

        # imu
        self.imu.eval_init(self.config.real_joint_dofs, sim.x0.ndofs)
        self.stdev_na = np.array(self.imu.stdev_na)
        self.stdev_nom = np.array(self.imu.stdev_nom)

        # noise: process
        Q = np.eye(self.num_noise)
        Q[0:3, 0:3] = self.dt**2 * self.stdev_na**2 * np.eye(3)
        Q[3:6, 3:6] = self.dt**2 * self.stdev_nom**2 * np.eye(3)
        Q[6:13, 6:13] = np.diag(self.config.process_noise_rw)
        self.Q = Q

        # noise: measurement
        self.R = np.diag(self.config.meas_noise)

        # buffer
        self._om_old = self.imu.om.squeeze()
        self._acc_old = self.imu.acc.squeeze()
        self.R_WB_old = self.states.q.rot

        # static matrices
        self.H = np.zeros([self.num_meas, self.num_error_states])
        self.H[0:6,18:24] = np.eye(6)
        self.H[6,15] = 1

        # plot
        self.traj = FilterTraj("kf")
        self.traj.append_propagated_states(self.config.min_t, self.states)

        # metrics
        self.mse = 0
        self.update_mse = 0 # non-cumulative; is overwritten every update stage

    def reset(self, x0, cov0, notch0):
        self.states = copy(x0)
        self.P = np.copy(cov0)
        self._x = []
        self._u = []
        self.dt = 0.

        # imu / noise
        self.imu.reset()
        self.imu.eval_init(self.config.real_joint_dofs, notch0)

        self.traj.reset()
        self.traj.append_propagated_states(self.config.min_t, self.states)
        self.mse = 0

    def update_noise_matrices(self):
        Q = np.eye(self.num_noise)
        Q[0:3, 0:3] = self.dt**2 * self.stdev_na**2 * np.eye(3)
        Q[3:6, 3:6] = self.dt**2 * self.stdev_nom**2 * np.eye(3)
        Q[6:13, 6:13] = np.diag(self.config.process_noise_rw)
        self.Q = Q

        self.R = np.diag(self.config.meas_noise)

    @property
    def config(self):
        return self.sim.config
    @config.setter
    def config(self, obj):
        self.sim.config = obj

    @property
    def run_id(self):
        return self._run_id

    @run_id.setter
    def run_id(self, val):
        self._run_id = val

    @property
    def x(self):
        self._x = self.states.vec
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
        return Quaternion(w=1., v=(0.5 * self.dt * self.om_old),
                do_normalise=True )

    @property
    def acc_old(self):
        return self._acc_old

    @acc_old.setter
    def acc_old(self, val):
        self._acc_old = val.squeeze()

    def run(self, camera, k, run_desc_str, verbose=True):
        """ Filter main loop (t>=1) over all camera frames,
            not counting IC.
            arg: k should already be adjusted to start at 1.
        """
        old_t           = self.config.min_t
        cam_timestamps  = tqdm(enumerate(camera.t[1:]),
                            total=camera.max_vals, initial=1,
                            desc=run_desc_str,
                            dynamic_ncols=True,
                            disable=not self.show_progress)
        self.run_id     = k

        for i, t in cam_timestamps:
            i_cam = i + 1 # not counting IC
            self.run_one_epoch(old_t, t, i_cam, camera)
            self.calculate_update_mse(i_cam, camera)

            if self.update_mse >= UPDATE_MSE_THRESHOLD:
                print(f'update_mse = {self.update_mse:.3E}')
            else:
                # only save states every ten frames and only if below threshold
                if (i_cam % SAVE_STATES_FRAME) == 0:
                    print(f'Saving states at frame {i_cam}')
                    self.save_states(i_cam, t)

            old_t = t

        # normalise dof_metric
        self.mse = self.calculate_metric(i_cam, camera)

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
        current_cam = camera.at_index(i_cam) # not counting IC
        ang_notch = camera.get_notch_at(i_cam)[0]
        self.update(t, current_cam, ang_notch)

    def propagate_imu(self, t0, tn):
        """ Generates IMU data between old and current camera frame,
            then uses this data to propagate the states as many times
            as there are IMU data between frames.
        """
        cam_queue = self.imu.cam.generate_queue(t0, tn)

        real_probe_dofs = self.config.real_joint_dofs

        old_ti = t0
        for ii, ti in enumerate(cam_queue.t):
            interp = cam_queue.at_index(ii)
            real_probe_dofs[0][6] = interp.notch
            real_probe_dofs[1][6] = interp.notch_d
            real_probe_dofs[2][6] = interp.notch_dd

            om, acc = self.imu.eval_expr_single(ti,
                *real_probe_dofs,
                interp.acc, interp.R,
                interp.om, interp.alp, overwrite=True)

            self.imu.ref.append_value(ti, interp.vec, interp.notch_arr)

            self.dt = ti - old_ti
            self.propagate(ti, om, acc)

            old_ti = ti

    def propagate(self, t, om, acc):
        self._predict_nominal(om, acc)
        self._predict_error()
        self._predict_error_covariance()

        # Buffer
        self.om_old = om
        self.acc_old = acc
        self.R_WB_old = self.states.q.rot

        # for plotting
        self.traj.append_propagated_states(t, self.states)

    def _predict_nominal(self, om, acc):
        est_probe = self.probe.get_est_fwkin(self.states.dofs,
                        self.states.ndofs)

        res = [casadi.DM(r).full() \
                    for r in eqns.f_predict(self.dt,
                        *self.x,
                        *self.u,
                        *est_probe,
                        om, acc)]

        self.states.set(res)

    def _predict_error(self):
        """ Calculates jacobian of the error state kinematics w.r.t. error states and w.r.t. noise. """

        Fx = casadi.SX.eye(self.num_error_states)
        Fx[0:3, 3:6] = self.dt * casadi.SX.eye(3)
        Fx[3:6, 6:9] = - self.R_WB_old @ casadi.skew(self.acc_old) * self.dt
        Fx[6:9, 6:9] = self.Om_old.rot.T
        # Fx[9:15, :] # dofs
        Fx[15:17, 16:18] = Fx[15:17, 16:18] + self.dt * casadi.SX.eye(2)

        self.Fx = self._cam_error_jacobian(Fx, syms.err_x)

        # motion model noise jacobian
        Fi = casadi.SX.zeros(self.num_error_states, self.num_noise)
        Fi[3:15, 0:12] = casadi.SX.eye(12)
        Fi[17, 12] = 1
        self.Fi = self._cam_error_jacobian(Fi, syms.n)

        """ returns zero
        self.err_states.set(self.Fx @ self.err_states.vec) """

    def _cam_error_jacobian(self, jac, vars_wrt):
        """ Fills the error jacobian (either w.r.t. error state or
            w.r.t. noise) for the camera state entries. """

        err_p_C_next = syms.err_p_C \
                + syms.dt * syms.get_err_pc_dot(self.probe)
        err_theta_C_next = syms.err_theta_C \
                + syms.dt * syms.get_err_theta_c_dot(self.probe)

        l_in = 0
        r_in = 0
        for x in vars_wrt:
            r_in += x.shape[0]
            jac[18:21,l_in:r_in] = casadi.jacobian(err_p_C_next, x)
            jac[21:,l_in:r_in] = casadi.jacobian(err_theta_C_next, x)
            l_in = r_in

        fun_jac = casadi.Function('f_jac',
            [syms.dt, syms.dofs, syms.notchdofs,
                syms.err_dofs, syms.err_notchdofs,
                syms.R_WB, *syms.u,
                syms.n_a, syms.n_om, syms.n_dofs, syms.n_notch_acc,
                syms.err_theta, syms.err_theta_C], [jac],
            ['dt', 'dofs', 'notchdofs', 'err_dofs', 'err_notchdofs', 'R_WB',
                *syms.u_str,
                'n_a', 'n_om', 'n_dofs', 'n_notch_acc',
                'err_theta', 'err_theta_C'], ['jac']
            )
        return casadi.DM(
                fun_jac( dt         = self.dt,
                        dofs        = self.states.dofs,
                        notchdofs   = self.states.ndofs,
                        R_WB        = self.R_WB_old,
                        B_om_BW     = self.om_old,
                        B_acc_BW    = self.acc_old,
                        n_a         = self.stdev_na,
                        n_om        = self.stdev_nom,
                        n_dofs      = self.config.process_noise_rw_std[0:6],
                        n_notch_acc = self.config.process_noise_rw_std[6],
                        # error states have an expectation of zero
                        err_dofs    = casadi.DM.zeros(6,),
                        err_theta   = casadi.DM([0., 0., 0.]),
                        err_theta_C = casadi.DM([0., 0., 0.]),
                        err_notchdofs = casadi.DM([0., 0., 0.]),
                        )['jac']).full()

    def _predict_error_covariance(self):
        """ Calculates the predicted covariance.
            Q is calculated from IMU noise and
            from the DOF random walk params.
        """
        self.P = self.Fx @ self.P @ self.Fx.T + self.Fi @ self.Q @ self.Fi.T

    def update(self, t, camera, ang_notch):
        # compute gain        
        H = self.H

        S = H @ self.P @ H.T + self.R # 7x7
        try:
            K = self.P @ H.T @ np.linalg.inv(S) # 24x7
        except np.linalg.LinAlgError as e:
            print(f"ERROR: {e}!")
            print("Stopping simulation.")
            return None

        # correct virtual SLAM reading to physical SLAM
        notch_quat = Quaternion(val=np.array([0, 0, ang_notch]), euler='xyz')
        cam_rot_corrected = notch_quat * camera.qrot
        # cam_rot_corrected = camera.qrot

        # compute error state
        res_p_cam = camera.pos.reshape((3,)) - self.states.p_cam.reshape((3,))
        err_q = cam_rot_corrected.conjugate * self.states.q_cam
        res_q_cam = err_q.angle * err_q.axis
        res_notch = ang_notch - self.states.ndofs[0]
        # res_notch = 0 - self.states.ndofs[0]

        res = np.hstack((res_p_cam, res_q_cam, res_notch))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self.states.apply_correction(err)
        I = np.eye(self.num_error_states)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        # reset error states
        G = np.eye(self.num_error_states) # 24x24
        G[6:9, 6:9] = np.eye(3) - skew(0.5 * err.theta)
        G[-3:, -3:] = np.eye(3) - skew(0.5 * err.theta_c)
        self.P = G @ self.P @ G.T

        # for plotting
        self.traj.append_updated_states(t, self.states)

        return K

    def get_upd_values(self, obj, comp, indices):
        """ returns obj (e.g. self.traj or self.imu.ref) component
            at update timestamps only. """
        if indices == -1:
            return obj.__dict__[comp][-1]
        else:
            return np.array([obj.__dict__[comp][i] for i in indices])

    def calculate_update_mse(self, i_cam, camera):
        cam_reference = camera.rotated if camera.rotated else camera
        current_cam = cam_reference.at_index(i_cam)

        sum_res_cam = 0
        for compc in self.traj.labels_camera[0:6]:
            abs_err = np.array(cam_reference.traj.__dict__[compc[:-1]][i_cam]) \
                - self.get_upd_values(self.traj, compc, -1)
            sum_res_cam += np.square(abs_err)

        sum_res_imu = 0
        for comp in self.traj.labels_imu[3:9]:
            abs_err = self.get_upd_values(self.traj, comp, -1) \
                - self.get_upd_values(self.imu.ref, comp, -1)
            sum_res_imu += np.square(abs_err)

        num_states = 12
        update_mse = (sum_res_cam + sum_res_imu) / num_states

        self.update_mse = update_mse

    def calculate_metric(self, i_cam, camera):
        cam_reference = camera.rotated if camera.rotated else camera

        indices_range = range(0, len(self.traj.t), self.config.num_interframe_vals)
        upd_indices = [x for x in indices_range]

        sum_res_cam = 0
        for compc in self.traj.labels_camera[0:6]:
            # -1: remove 'c' from string';
            # i_cam + 1: include final value before breaking simulation
            abs_err = np.array(cam_reference.traj.__dict__[compc[:-1]][:i_cam + 1]) \
                - self.get_upd_values(self.traj, compc, upd_indices)
            sum_res_cam += np.sum(np.square(abs_err))

        sum_res_imu = 0
        for comp in self.traj.labels_imu[3:9]:
            abs_err = self.get_upd_values(self.traj, comp, upd_indices[:-1]) \
                - self.get_upd_values(self.imu.ref, comp, upd_indices[:-1]) # no imu ref calculated at end
            sum_res_imu += np.sum(np.square(abs_err))

        num_states = 12
        num_frames = len(upd_indices)
        avg_sum_res = (sum_res_cam + sum_res_imu) / (num_states * num_frames) \
            + self.calculate_dof_metric()

        return avg_sum_res

    def calculate_dof_metric(self):
        """ at end of run """
        res = self.states.dofs - self.config.real_imu_dofs
        return np.dot(res, res) / 6

    def save(self):
        self.config.mse = self.mse
        # self.config.save('./configs.txt')
        self.traj.write_to_file(self.config.traj_kf_filepath)
        self.imu.ref.write_to_file(self.config.traj_imuref_filepath)

    def save_states(self, i_cam, t):
        # will be the new initial states
        self.old_states = deepcopy(self.states)
        self.old_P = np.copy(self.P)

        self.old_i_cam = i_cam
        self.old_t = t

        self.old_om_old = np.copy(self.om_old)
        self.old_acc_old = np.copy(self.acc_old)
        self.old_R_WB_old = np.copy(self.R_WB_old)

    def reset_warm(self):
        # set new initial states
        self.states = deepcopy(self.old_states)
        self.P = np.copy(self.old_P)

        # reset camera input to old time
        old_t           = self.old_t
        new_cam_timestamps  = tqdm(enumerate(camera.t[self.old_i_cam + 1:]), # no IC
                            total=camera.max_vals, initial=self.old_i_cam + 1,
                            desc='Warm start',
                            dynamic_ncols=True,
                            disable=not self.show_progress)

        # new camera object
        # new imu object referencing new camera

        # buffer
        self.om_old = np.copy(self.old_om_old)
        self.acc_old = np.copy(self.old_acc_old)
        self.R_WB_old = np.copy(self.old_R_WB_old)

        # initial value for imu
        self.imu.eval_init(self.config.real_joint_dofs, self.states.ndofs,
            overwrite=True)

        # reset imu input to old time

        # delete trajectory up till old time
        # append initial values

        return new_cam_timestamps

    def plot(self, camera, compact):
        if not self.config.do_plot:
            return

        t_end = self.traj.t[-1]
        plotter = FilterPlot(self, camera)
        if compact:
            plotter.plot_compact(t_end)
        else:
            plotter.plot(t_end)