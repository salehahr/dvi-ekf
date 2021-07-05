from copy import copy
from math import factorial
import numpy as np

from .Quaternion import Quaternion, skew
from .Trajectory import VisualTraj
from .States import States, ErrorStates

import casadi

Fi = np.zeros([9, 6])
Fi[3:, :] = np.eye(6)  # motion model noise jacobian

H_x = np.zeros([7, 10])
H_x[:3,:3] = np.eye(3)
H_x[3:,-4:] = np.eye(4) # part of measurement model jacobian

class Filter(object):
    def __init__(self, imu, IC, P0, num_meas, num_control):
        self.num_states = IC.size
        self.num_error_states = IC.size - 1
        self.num_meas = num_meas
        self.num_control = num_control

        self.states = copy(IC)
        self._x = []
        self._u = []

        self.dt = 0.
        self.traj = VisualTraj("kf")

        # imu
        self.imu = imu
        self.probe = self.imu.probe

        # buffer
        self._om_old = imu.om.squeeze()
        self._acc_old = imu.acc.squeeze()
        self.R_WB_old = self.states.q.rot

        # covariance
        self.P = P0

        # symbolic states, inputs
        p = casadi.SX.sym('p_B', 3)
        v = casadi.SX.sym('v_B', 3)
        R_WB = casadi.SX.sym('R_WB', 3,3)
        dofs = casadi.SX.sym('dofs', 6)
        p_BC = casadi.SX.sym('p_BC', 3)

        x_sym = [p, v, R_WB, dofs, p_BC]
        x_str = ['p', 'v', 'R_WB', 'dofs', 'p_BC']

        acc = casadi.SX.sym('acc', 3)
        om = casadi.SX.sym('om', 3)

        self._u = []
        u_sym = [om, acc]
        u_str = ['om', 'acc']

        # symbolic model equations
        dt = casadi.SX.sym('dt')
        p_next = p + dt * v + dt**2 / 2 * R_WB @ acc

        self.fun_nominal = casadi.Function('f_nom',
            [dt, *x_sym, *u_sym],
            [   p_next,
                v + dt * R_WB @ acc,
                R_WB + R_WB @ casadi.skew(dt * om),
                dofs,
                p_next + R_WB @ p_BC ], # TODO: p_BC as a function of DOFs
            ['dt', *x_str, *u_str],
            ['p_next', 'v_next', 'R_WB_next', 'dofs_next', 'p_cam_next'])

        # error states
        dx_sym = casadi.SX.sym('dx', self.num_error_states)

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
    def jac_X_deltx(self):
        x, y, z, w = self.states.q.xyzw
        Q_deltth = 0.5 * np.array([[-x, -y, -z],
                                   [ w, -z,  y],
                                   [ z,  w, -x],
                                   [-y,  x,  w]])

        X_deltx = np.zeros([10, 9])
        X_deltx[:6,:6] = np.eye(6)
        X_deltx[-4:,-3:] = Q_deltth

        return X_deltx

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
        res = [casadi.DM(r).full() \
                    for r in self.fun_nominal(self.dt, *self.x, *self.u)]

        self.states.p = res[0]
        self.states.v = res[1]
        self.states.q = res[2]
        self.states.dofs = res[3]
        self.states.p_cam = res[4]

    def _predict_error(self):
        F = np.eye(9)
        F[0:3, 3:6] = self.dt * np.eye(3)
        F[3:6, 6:9] = - self.R_WB_old @ skew(self.acc_old) * self.dt
        F[6:9, 6:9] = self.Om_old.rot.T
        self.Fx = F

        # returns zero
        # self.err_states = self.Fx @ self.err_states

    def _predict_error_covariance(self):
        Q = (self.dt ** 2) * self.Qc # integrate acceleration to obstain position
        self.P = self.Fx @ self.P @ self.Fx.T + Fi @ Q @ Fi.T

    def update(self, camera):
        # compute gain        
        H = H_x @ self.jac_X_deltx
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # compute error state
        res_p = camera.pos.reshape((3,)) - self.states.p.reshape((3,))
        res_q = (camera.qrot - self.states.q).xyzw
        res = np.hstack((res_p, res_q))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self.states.apply_correction(err)
        self.P = (np.eye(9) - K @ H) @ self.P