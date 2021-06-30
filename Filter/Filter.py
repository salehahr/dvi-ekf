from math import factorial
import numpy as np

from .Quaternion import Quaternion
from .Trajectory import VisualTraj

Fi = np.zeros([9, 6])
Fi[3:, :] = np.eye(6)  # motion model noise jacobian

H_x = np.zeros([7, 10])
H_x[:3,:3] = np.eye(3)
H_x[3:,-4:] = np.eye(4) # part of measurement model jacobian

def skew(x):
    return np.array([[0,    -x[2], x[1]],
                     [x[2],    0, -x[0]],
                     [-x[1], x[0],    0]])

class States(object):
    def __init__(self, p, v, q):
        self.p = np.asarray(p)
        self.v = np.asarray(v)
        self.q = Quaternion(xyzw=q, do_normalise=True)

        self.size = len(p) + len(v) + 4

    def apply_correction(self, err):
        self.p += err.dp.reshape(3,1)
        self.v += err.dv.reshape(3,1)
        self.q = self.q * err.dq
        self.q.normalise()

class ErrorStates(object):
    def __init__(self, vec):
        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]

        self.dp = np.asarray(p)
        self.dv = np.asarray(v)
        self.dq = Quaternion(v=theta/2, w=1.)

class Filter(object):
    def __init__(self, imu, IC, P0, num_meas, num_control):
        self.num_states = IC.size
        self.num_error_states = IC.size - 1
        self.num_meas = num_meas
        self.num_control = num_control

        self.dt = 0.
        self.traj = VisualTraj("kf")

        # states
        self.states = IC

        # imu
        self.imu = imu
        self._om_old = None
        self._acc_old = None

        # covariance
        self.P = P0

    @property
    def om_old(self):
        return self._om_old
    @property
    def acc_old(self):
        return self._acc_old

    @om_old.setter
    def om_old(self, val):
        if isinstance(val, np.ndarray):
            val = val.squeeze()
        self._om_old = val
    @acc_old.setter
    def acc_old(self, val):
        if isinstance(val, np.ndarray):
            val = val.squeeze()
        self._acc_old = val

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

    def init_imu(self, *joint_dofs0):
        self.imu.eval_init(*joint_dofs0)

        self.om_old = self.imu.om
        self.acc_old = self.imu.acc

    def propagate(self, t, om, acc, do_prop_only=False):
        self._predict_nominal()
        self._predict_error()
        self._predict_error_covariance()

        # IMU buffer
        self.om_old = om
        self.acc_old = acc

        # for plotting
        self.traj.append_state(t, self.states)

    def _predict_nominal(self):
        self.R_WB_old = self.states.q.rot
        W_acc_B_old = (self.R_WB_old @ self.acc_old).reshape(3,1)

        # position p
        self.states.p = self.states.p \
                + self.dt * self.states.v \
                + self.dt**2/2. * W_acc_B_old

        # velocity v
        self.states.v = self.states.v \
                + W_acc_B_old * self.dt

        # orientation q: Solà
        Om = Quaternion(w=1., v=(0.5 * self.dt * self.om_old) )
        self.states.q = self.states.q * Om

        # # orientation q: Kok
        # Om = Quaternion(w=0., v=(0.5 * self.dt * self.om_old) )
        # self.states.q += Om * self.states.q

        self.states.q.normalise()

        self.Om_old = Om

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