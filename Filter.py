from math import factorial
import numpy as np
import quaternion

def skew(x):
    return np.array([[0,    -x[2], x[1]],
                     [x[2],    0, -x[0]],
                     [-x[1], x[0],    0]])

class States(object):
    def __init__(self, p, v, q, bw, ba, scale, p_offset, q_offset):
        self.p = np.asarray(p)
        self.v = np.asarray(v)
        self.q = q.normalized()

        self.bw = np.asarray(bw)
        self.ba = np.asarray(ba)
        self.scale = scale

        self.p_offset = np.asarray(p_offset)
        self.q_offset = np.asarray(q_offset.normalized())

        self.size = len(p) + len(v) + 4 \
                + len(bw) + len(ba) + 1 \
                + len(p_offset) + 4

class Filter(object):
    def __init__(self, num_states, num_meas, num_control):
        self.num_states = num_states
        self.num_error_states = num_states - 2
        self.num_meas = num_meas
        self.num_control = num_control

        self.dt = None

        # states
        self.states = None
        self.p = None
        self.v = None
        self.q = None
        self.bw = None
        self.ba = None
        self.scale = None
        self.p_offset = None
        self.q_offset = None

        self.p_VW = np.asarray([0., 0., 0.])
        self.q_VW = np.quaternion(1., 0., 0., 0.)

        # imu
        self.om_old = None
        self.acc_old = None

        # covariance
        self.p = None

    def set_states(self, states):
        self.states = states

        self.p = states.p
        self.v = states.v
        self.q = states.q
        self.bw = states.bw
        self.ba = states.ba
        self.scale = states.scale
        self.p_offset = states.p_offset
        self.q_offset = states.q_offset

    def set_covariance(self, cov_matr):
        self.P = cov_matr

    def propagate_states(self, imu):
        v_old = self.v
        R_WB_old = quaternion.as_rotation_matrix(self.q)

        om = imu.om - self.bw
        om_q = np.quaternion(0., om[0], om[1], om[2])

        self.q += self.dt / 2. * om_q * self.q # quat multiplication
        self.q = self.q.normalized()
        R_WB = quaternion.as_rotation_matrix(self.q)

        self.v += self.dt / 2. * ( \
            R_WB_old @ (self.acc_old - self.ba) \
            + R_WB @ (imu.acc - self.ba) )
        self.p += self.dt / 2. * (v_old + self.v)

        self._update_states()
        self.om_old = imu.om
        self.acc_old = imu.acc

    def propagate_covariance(self, imu):
        om = imu.om - self.bw
        acc = imu.acc - self.ba

        Qda, Qdb, Qdc, Fda, Fdb = self._calculate_Qd(om, acc)

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

    def update(self, camera):
        p_VC = camera.pos
        q_VC = camera.qrot

        R_VW = quaternion.as_rotation_matrix(self.q_VW)
        R_WB = quaternion.as_rotation_matrix(self.q)
        R_BC = quaternion.as_rotation_matrix(self.q_offset)

        h_scale = R_VW @ (R_WB @ self.p_offset + self.p)
        h_scale = np.reshape(h_scale, (h_scale.shape[0], -1))

        zero3 = np.zeros((3, 3))
        Hp = np.hstack((
            R_VW * self.scale,
            zero3,
            -R_VW @ R_WB @ skew(self.p_offset) * self.scale,
            zero3,
            zero3,
            h_scale,
            R_VW @ R_WB * self.scale,
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

    def _calculate_Fd(self, om, acc):
        Fd = np.eye(self.num_error_states, self.num_error_states)
        R_WB = quaternion.as_rotation_matrix(self.q)

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
        R_WB = quaternion.as_rotation_matrix(self.q)

        Gc[3:6, 0:3] = -R_WB
        Gc[6:9, 6:9] = -np.eye(3, 3)
        Gc[9:12, -3:] = +np.eye(3, 3)
        Gc[12:15, 3:6] = +np.eye(3, 3)

        return Gc

    def _calculate_Qd(self, om, acc):
        stdev_na = [0.1, 0.1, 0.1]
        stdev_nba = [0.1, 0.1, 0.1]
        stdev_nw = [0.05, 0.05, 0.05]
        stdev_nbw = [0.1, 0.1, 0.1]
        stdevs = np.hstack((stdev_na, stdev_nba, \
                    stdev_nw, stdev_nbw))
        Qc = np.square(np.diag(stdevs))

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

    def _update_states(self):
        self.states.p = self.p
        self.states.v = self.v
        self.states.q = self.q
        self.states.bw = self.bw
        self.states.ba = self.ba
        self.states.scale = self.scale
        self.states.p_offset = self.p_offset
        self.states.q_offset = self.q_offset