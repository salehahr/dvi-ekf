from math import factorial
import numpy as np

from .Quaternion import Quaternion
from .Trajectory import VisualTraj

l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

def skew(x):
    return np.array([[0,    -x[2], x[1]],
                     [x[2],    0, -x[0]],
                     [-x[1], x[0],    0]])

class States(object):
    def __init__(self, p, v, q, bw, ba):#, scale, p_offset, q_offset):
        self.p = np.asarray(p)
        self.v = np.asarray(v)
        self.q = Quaternion(xyzw=q, do_normalise=True)

        self.bw = np.asarray(bw)
        self.ba = np.asarray(ba)
        # self.scale = scale

        # self.p_offset = np.asarray(p_offset)
        # self.q_offset = Quaternion(xyzw=q_offset, do_normalise=True)

        self.size = len(p) + len(v) + 4 \
                + len(bw) + len(ba) #+ 1 \
                # + len(p_offset) + 4

    def apply_correction(self, err):
        self.p += err.dp
        self.v += err.dv
        self.q = err.dq * self.q
        self.q.normalise()

        self.bw += err.dbw
        self.ba += err.dba
        # self.scale += err.dscale

        # self.p_offset += err.dp_offset
        # self.q_offset = err.dq_offset * self.q_offset
        # self.q_offset.normalise()

class ErrorStates(object):
    def __init__(self, vec):
        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]
        bw = vec[9:12]
        ba = vec[12:15]
        # scale = vec[15]
        # p_offset = vec[16:19]
        # theta_offset = vec[19:]

        self.dp = np.asarray(p)
        self.dv = np.asarray(v)
        self.dq = Quaternion(v=theta/2, w=1.)

        self.dbw = np.asarray(bw)
        self.dba = np.asarray(ba)
        # self.dscale = scale

        # self.dp_offset = np.asarray(p_offset)
        # self.dq_offset = Quaternion(v=theta_offset/2, w=1.)

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

        self.p_VW = np.asarray([0., 0., 0.])
        self.q_VW = Quaternion(xyzw=[0., 0., 0., 1.])

        # imu
        self.om_old = None
        self.acc_old = None

        # covariance
        self.P = P0

    def propagate(self, t, imu, Qc, do_prop_only=False):
        self.propagate_states(imu, Qc)

        # if not do_prop_only:
            # self.propagate_covariance(imu, Qc)

        self.traj.append_state(t, self.states)

    def propagate_states(self, imu, Qc):
        v_old = self.states.v
        R_WB_old = self.states.q.rot

        # orientation q
        om = Quaternion(w=0., v=(imu.om - self.states.bw) )
        self.states.q += self.dt / 2. * om * self.states.q
        self.states.q.normalise()

        # velocity v (both eqns are equiv)
        self.states.v += R_WB_old @ (self.acc_old - self.states.ba) \
                            * self.dt

        # position p (both eqns are equiv)
        self.states.p += \
            self.dt * v_old \
            + self.dt**2/2. \
            * (R_WB_old @ (self.acc_old - self.states.ba))

        F = np.eye(9)
        F[0:3, 3:6] = self.dt * np.eye(3)
        F[3:6, 6:9] = - R_WB_old @ skew(self.acc_old - self.states.ba) \
                            * self.dt
        F[6:9, 6:9] = om.rot.T
        self.Fx = F

        Qc = (self.dt ** 2) * Qc # integration acceleration to obstain position
        self.P = F @ self.P @ F.T + l_jac @ Qc @ l_jac.T

        self.om_old = imu.om
        self.acc_old = imu.acc

    def propagate_covariance(self, imu, Qc):
        om = imu.om - self.states.bw
        acc = imu.acc - self.states.ba

        Qda, Qdb, Qdc, Fda, Fdb = self._calculate_Qd(Qc, om, acc)

        P = self.P
        PC = P[0:9, 0:9]
        PD = P[0:9, 9:15]
        PE = P[0:9, 15:]
        PG = P[9:15, 9:15]
        PH = P[9:15, -7:]

        # PC
        P[0:9, 0:9] = Fda@PC@Fda.T + Fdb@PD.T@Fda.T + Fda@PD@Fdb.T \
                + Fdb@PG@Fdb.T + Qda

        # PD
        PD = Fda@PD + Fdb@PG + Qdb
        P[0:9, 9:15] = PD
        P[9:15, 0:9] = PD.T

        # PE
        P[0:9, 15:] = Fda@PE + Fdb@PH
        P[0:9, 15:] = PE
        P[-7:, 0:9] = PE.T

        # PG
        P[9:15, 9:15] += Qdc

        self.P = P
        # self.P = self.Fd @ self.P @ self.Fd.T + Qd

    def update(self, camera, R):
        p_VC = camera.pos
        q_VC = camera.qrot

        R_VW = self.q_VW.rot
        R_WB = self.states.q.rot
        R_BC = self.states.q_offset.rot

        h_scale = R_VW @ (R_WB @ self.states.p_offset + self.states.p)
        h_scale = np.reshape(h_scale, (h_scale.shape[0], -1))

        zero3 = np.zeros((3, 3))
        Hp = np.hstack((
            R_VW * self.states.scale,
            zero3,
            -R_VW @ R_WB @ skew(self.states.p_offset) * self.states.scale,
            zero3,
            zero3,
            h_scale,
            R_VW @ R_WB * self.states.scale,
            zero3,
            # np.eye(3, 3) * self.scale,
            # -R_VW @ skew( self.p_offset + R_WB @ self.p_offset ) \
                # * self.scale
        ))

        Hq = np.hstack((
            zero3,
            zero3,
            R_BC,
            zero3,
            zero3,
            np.zeros((3, 1)),
            zero3,
            np.eye(3, 3),
            # zero3,
            # C_q_c_i*C_q_i_w;
        ))

        H = np.vstack((Hp, Hq))

        # residual
    # def _calculate_residual(self, p_vc):
        z_est = ( R_VW @ (self.states.p + R_WB @ self.states.p_offset) + self.p_VW ) * self.states.scale
        r_p = p_VC - z_est;

        zq = self.states.q_offset * self.states.q * self.q_VW

        r_q = (q_VC * ( zq ).conjugate ).conjugate.wxyz

        r = np.hstack((
            r_p,
            2*r_q[1],
            2*r_q[2],
            2*r_q[3],
        ))

        # calculate Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        x_error = ErrorStates(K @ r)

        # apply Kalman gain
        self.states.apply_correction(x_error)

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