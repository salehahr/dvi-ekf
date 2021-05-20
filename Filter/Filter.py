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
        self.p += err.dp
        self.v += err.dv
        self.q = err.dq * self.q
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
    def __init__(self, IC, P0, num_meas, num_control):
        self.num_states = IC.size
        self.num_error_states = IC.size - 1
        self.num_meas = num_meas
        self.num_control = num_control

        self.dt = 0.
        self.traj = VisualTraj("kf")

        # states
        self.states = IC

        # imu
        self.om_old = None
        self.acc_old = None

        # covariance
        self.P = P0

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

    def propagate(self, t, imu, Qc, do_prop_only=False):
        self._predict_nominal()
        self._predict_error()
        self._predict_error_covariance(Qc)

        self.om_old = imu.om
        self.acc_old = imu.acc

        self.traj.append_state(t, self.states)

    def _predict_nominal(self):
        self.R_WB_old = self.states.q.rot

        # position p
        self.states.p += \
            self.dt * self.states.v \
            + self.dt**2/2. * self.R_WB_old @ self.acc_old

        # velocity v
        self.states.v += self.R_WB_old @ self.acc_old * self.dt

        # orientation q: Solà
        Om = Quaternion(w=1., v=(0.5 * self.dt * self.om_old) ) #works but the 0.5 wasn't in the paper.....
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

    def _predict_error_covariance(self, Qc):
        Qc = (self.dt ** 2) * Qc # integrate acceleration to obstain position
        self.P = self.Fx @ self.P @ self.Fx.T + Fi @ Qc @ Fi.T

    def update(self, camera, R):
        # compute gain        
        H = H_x @ self.jac_X_deltx
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # compute error state
        res_p = camera.pos.reshape((3,)) - self.states.p.reshape((3,))
        res_q = (camera.qrot - self.states.q).xyzw
        res = np.hstack((res_p, res_q))
        err = ErrorStates(K @ res)

        # correct predicted state and covariance
        self.states.apply_correction(err)
        self.P = (np.eye(9) - K @ H) @ self.P

        # p_VC = camera.pos
        # q_VC = camera.qrot

        # R_VW = self.q_VW.rot
        # R_WB = self.states.q.rot
        # R_BC = self.states.q_offset.rot

        # h_scale = R_VW @ (R_WB @ self.states.p_offset + self.states.p)
        # h_scale = np.reshape(h_scale, (h_scale.shape[0], -1))

        # zero3 = np.zeros((3, 3))
        # Hp = np.hstack((
            # R_VW * self.states.scale,
            # zero3,
            # -R_VW @ R_WB @ skew(self.states.p_offset) * self.states.scale,
            # zero3,
            # zero3,
            # h_scale,
            # R_VW @ R_WB * self.states.scale,
            # zero3,
            # # np.eye(3, 3) * self.scale,
            # # -R_VW @ skew( self.p_offset + R_WB @ self.p_offset ) \
                # # * self.scale
        # ))

        # Hq = np.hstack((
            # zero3,
            # zero3,
            # R_BC,
            # zero3,
            # zero3,
            # np.zeros((3, 1)),
            # zero3,
            # np.eye(3, 3),
            # # zero3,
            # # C_q_c_i*C_q_i_w;
        # ))

        # H = np.vstack((Hp, Hq))

        # # residual
    # # def _calculate_residual(self, p_vc):
        # z_est = ( R_VW @ (self.states.p + R_WB @ self.states.p_offset) + self.p_VW ) * self.states.scale
        # r_p = p_VC - z_est;

        # zq = self.states.q_offset * self.states.q * self.q_VW

        # r_q = (q_VC * ( zq ).conjugate ).conjugate.wxyz

        # r = np.hstack((
            # r_p,
            # 2*r_q[1],
            # 2*r_q[2],
            # 2*r_q[3],
        # ))

        # # calculate Kalman gain
        # S = H @ self.P @ H.T + R
        # K = self.P @ H.T @ np.linalg.inv(S)
        # x_error = ErrorStates(K @ r)

        # apply Kalman gain
        # self.states.apply_correction(x_error)

    def _calculate_Fd(self, om, acc):
        Fd = np.eye(self.num_error_states, self.num_error_states)
        R_WB = self.states.q.rot

        dt = self.dt * np.eye(3, 3)
        delt2 = self.dt**2/2 * np.eye(3, 3)
        delt3 = self.dt**3/factorial(3) * np.eye(3, 3)
        delt4 = self.dt**4/factorial(4) * np.eye(3, 3)
        delt5 = self.dt**5/factorial(5) * np.eye(3, 3)

        acc_sk = skew(acc)
        om_sk = skew(om)
        om_sk_sq = om_sk @ om_sk

        A = -R_WB @ acc_sk @ ( delt2 - delt3 * om_sk + delt4 * om_sk_sq )
        B = -R_WB @ acc_sk @ ( delt3 + delt4 * om_sk - delt5 * om_sk_sq )
        C = -R_WB @ acc_sk @ ( dt    - delt2 * om_sk + delt3 * om_sk_sq )
        D = -A
        E = np.eye(3, 3) - dt * om_sk + delt2 * om_sk_sq
        F = -dt + delt2 * om_sk - delt3 * om_sk_sq

        Fd[0:3, 3:6] = dt
        Fd[0:3, 6:9] = A
        Fd[0:3, 9:12] = B
        Fd[0:3, 12:15] = -R_WB @ delt2

        Fd[3:6, 6:9] = C
        Fd[3:6, 9:12] = D
        Fd[3:6, 12:15] = -R_WB @ dt

        Fd[6:9, 6:9] = E
        Fd[6:9, 9:12] = F

        self.Fd = Fd

        return Fd

    def _calculate_Gc(self):
        Gc = np.zeros((self.num_states, self.num_control))
        R_WB = self.states.q.rot

        Gc[3:6, 0:3] = -R_WB
        Gc[6:9, 6:9] = -np.eye(3, 3)
        Gc[9:12, -3:] = +np.eye(3, 3)
        Gc[12:15, 3:6] = +np.eye(3, 3)

        return Gc

    def _calculate_Qd(self, Qc, om, acc):
        stdevs = np.sqrt(np.diag(Qc))
        stdev_na = stdevs[0:3]
        stdev_nba = stdevs[3:6]
        stdev_nw = stdevs[6:9]
        stdev_nbw = stdevs[9:12]

        Nce = np.zeros((9, 9))
        Nch = np.zeros((6, 6))

        Nce[3:6, 3:6] = np.square(np.diag(stdev_na))
        Nce[6:9, 6:9] = np.square(np.diag(stdev_nw))

        Nch[0:3, 0:3] = np.square(np.diag(stdev_nbw))
        Nch[3:6, 3:6] = np.square(np.diag(stdev_nba))

        Fd = self._calculate_Fd(om, acc)
        Gc = self._calculate_Gc()

        Fda = Fd[0:9, 0:9]
        Fdb = Fd[0:9, 9:15]

        Qda = self.dt / 2. * (Fda @ Nce @ Fda.T + Fdb @ Nch @ Fdb.T + Nce)
        Qdb = self.dt / 2. * (Fdb @ Nch)
        Qdc = self.dt * Nch

        Qd_theirs_top = np.hstack((Qda, Qdb))
        Qd_theirs_bottom = np.hstack((Qdb.T, Qdc))
        Qd_theirs = np.vstack((Qd_theirs_top, Qd_theirs_bottom))

        # Qd_mine = self.dt * Fd @ Gc @ Qc @ Gc.T @ Fd.T

        # print(Qda)
        # print(Qd_mine[0:9, 0:9])
        # input()

        # Qd = Qd_mine
        Qd = Qd_theirs

        return Qda, Qdb, Qdc, Fda, Fdb