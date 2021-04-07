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

    def _update_states(self):
        self.states.p = self.p
        self.states.v = self.v
        self.states.q = self.q
        self.states.bw = self.bw
        self.states.ba = self.ba
        self.states.scale = self.scale
        self.states.p_offset = self.p_offset
        self.states.q_offset = self.q_offset