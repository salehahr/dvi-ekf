from copy import copy
from math import factorial
import numpy as np

from .Quaternion import Quaternion, skew
from .Trajectory import FilterTraj
from .States import States, ErrorStates

import casadi
from .symbols import *

class Filter(object):
    def __init__(self, imu, IC, P0):
        self.num_states = IC.size
        self.num_error_states = IC.size - 1
        self.num_meas = 7

        self.states = copy(IC)
        self._x = []
        self._u = []

        self.dt = 0.
        self.traj = FilterTraj("kf")

        # imu
        self.imu = imu
        self.probe = self.imu.probe

        # buffer
        self._om_old = imu.om.squeeze()
        self._acc_old = imu.acc.squeeze()
        self.R_WB_old = self.states.q.rot

        # covariance
        self.P = P0

    @property
    def x(self):
        self._x = [self.states.p,
                    self.states.v,
                    self.states.q.rot,
                    self.states.dofs,
                    self.states.p_cam]
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
        return Quaternion(w=1., v=(0.5 * self.dt * self.om_old) )

    @property
    def acc_old(self):
        return self._acc_old

    @acc_old.setter
    def acc_old(self, val):
        self._acc_old = val.squeeze()

    @property
    def Hx(self):
        Hx = np.zeros([7, self.num_states])
        Hx[:3,-3:] = np.eye(3)
        # Hx[6:10,-4:] = np.eye(4) # part of measurement model jacobian # TODO
        return Hx

    @property
    def jac_X_deltx(self):
        x, y, z, w = self.states.q.xyzw
        Q_deltth = 0.5 * np.array([[-x, -y, -z],
                                   [ w, -z,  y],
                                   [ z,  w, -x],
                                   [-y,  x,  w]])

        X_deltx = np.zeros([self.num_states, self.num_error_states])
        X_deltx[:6,:6] = np.eye(6)
        X_deltx[-4:,-3:] = Q_deltth

        return X_deltx

    @property
    def Fi(self):
        # motion model noise jacobian
        Fi = np.zeros([self.num_error_states, 12])
        Fi[3:15, :12] = np.eye(12)
        return Fi

    def propagate(self, t, om, acc, do_prop_only=False):
        self._predict_nominal()
        self._predict_error()
        self._predict_error_covariance()

        # Buffer
        self.om_old = om
        self.acc_old = acc
        self.R_WB_old = self.states.q.rot

        # for plotting
        self.traj.append_state(t, self.states)

    def _predict_nominal(self):
        p_next = p_B + dt * v_B + dt**2 / 2 * R_WB @ acc

        self.fun_nominal = casadi.Function('f_nom',
            [dt, *x, *u],
            [   p_next,
                v_B + dt * R_WB @ acc,
                R_WB + R_WB @ casadi.skew(dt * om),
                dofs,
                p_next + R_WB @ self.probe.p ], # TODO: p_BC as a function of DOFs
            ['dt', *x_str, *u_str],
            ['p_B_next', 'v_B_next', 'R_WB_next',
                'dofs_next', 'p_C_next'])

        res = [casadi.DM(r).full() \
                    for r in self.fun_nominal(self.dt,
                        *self.x,
                        *self.u)]

        self.states.p = res[0]
        self.states.v = res[1]
        self.states.q = res[2]
        self.states.dofs = res[3]
        self.states.p_cam = res[4]

    def _predict_error(self):
        err_p_C_dot = get_err_pc_dot(self.probe)

        fun_error = casadi.Function('f_err',
            [dt, *err_x, *u, *n, R_WB],
            [   err_p_B + dt * err_v_B,
                err_v_B + dt * (-R_WB @ casadi.skew(acc) @ err_theta) + n_v,
                -casadi.cross(om, err_theta) + n_om,
                n_dofs,
                err_p_C + dt * err_p_C_dot ],
            ['dt', *err_x_str, *u_str, *n_str, 'R_WB'],
            ['err_p_B_next', 'err_v_B_next', 'err_theta_next',
                'err_dofs_next', 'err_p_C_next'])

        jac = fun_error.jac()
        res = jac(  dt   = self.dt,
                    om   = self.om_old,
                    acc  = self.acc_old,
                    R_WB = self.R_WB_old)

        F = np.eye(self.num_error_states)

        F[0:3, 3:6] = self.dt * np.eye(3)
        F[3:6, 6:9] = - self.R_WB_old @ skew(self.acc_old) * self.dt
        F[6:9, 6:9] = self.Om_old.rot.T
        F[9:15, :]  = 0 # dofs

        # jacobian of err_p_cam w.r.t. err_x
        l_in = 0
        r_in = 0
        for x in err_x_str:
            name = 'Derr_p_C_nextD' + x
            res_np = res[name].full()

            r_in += res_np.shape[1]
            F[-3:,l_in:r_in] = res_np
            l_in = r_in

        self.Fx = F

        # returns zero
        # self.err_states = self.Fx @ self.err_states

    def _predict_error_covariance(self):
        Q = np.eye(12)
        Q[:6,:6] = (self.dt ** 2) * self.Qc # integrate acceleration to obstain position

        # random walk of dofs
        sigma = 0.01
        mu = 0
        Q[-6:,-6:] = self.dt * (sigma * np.random.randn(6, 6) + mu)

        self.P = self.Fx @ self.P @ self.Fx.T + self.Fi @ Q @ self.Fi.T

    def update(self, camera):
        # compute gain        
        H = self.Hx @ self.jac_X_deltx # 7x18
        S = H @ self.P @ H.T + self.R # 7x7
        K = self.P @ H.T @ np.linalg.inv(S) # 18x7

        # compute error state
        R_BC = casadi.DM(self.probe.R).full()
        res_p_cam = camera.pos.reshape((3,)) - self.states.p_cam.reshape((3,))
        res_q = (camera.qrot - self.states.q * R_BC).xyzw
        res = np.hstack((res_p_cam, res_q))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self.states.apply_correction(err)
        self.P = (np.eye(self.num_error_states) - K @ H) @ self.P