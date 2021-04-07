import numpy as np

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