import numpy as np
from .Quaternion import Quaternion
import math

class States(object):
    def __init__(self, p, v, q, dofs, ndofs, p_cam, q_cam):
        self._p = np.asarray(p).reshape(3,)
        self._v = np.asarray(v).reshape(3,)
        self._q = Quaternion(val=q, do_normalise=True)
        self._dofs = np.array(dofs).reshape(6,)
        self._ndofs = np.array(ndofs).reshape(3,)
        self._p_cam = p_cam.reshape(3,)
        self._q_cam = Quaternion(val=q_cam, do_normalise=True)

        self.size = len(p) + len(v) + len(self.q.xyzw) \
                    + len(dofs) + len(ndofs) \
                    + len(p_cam) + len(self.q_cam.xyzw)
        assert(self.size == 26)

        self.frozen_dofs = [False] * 6

    def apply_correction(self, err):
        self.p += err.dp.reshape(3,)
        self.v += err.dv.reshape(3,)
        self.q = self.q * err.dq
        self.q.normalise()

        for i, fr in enumerate(self.frozen_dofs):
            if not fr:
                self._dofs[i] += err.ddofs[i]

        self.ndofs += err.dndofs.reshape(3,)
        self.p_cam += err.dpc.reshape(3,)
        self.q_cam = self.q_cam * err.dqc
        self.q_cam.normalise()

    def set(self, vec):
        self.p = vec[0].squeeze()
        self.v = vec[1].squeeze()
        self.q = vec[2].squeeze()

        for i, fr in enumerate(self.frozen_dofs):
            if not fr:
                self._dofs[i] = vec[3].squeeze()[i]

        self.ndofs = np.array(vec[4:7]).squeeze()
        self.p_cam = vec[7].squeeze()
        self.q_cam = vec[8].squeeze()

    def __repr__(self):
        return f'State: p_cam ({self._p_cam}), ...'

    @property
    def vec(self):
        return [self.p, self.v, self.q.rot,
                    self.dofs, self.ndofs,
                    self.p_cam, self.q_cam.rot]

    @property
    def p(self):
        return self._p.copy()

    @p.setter
    def p(self, val):
        self._p = val

    @property
    def v(self):
        return self._v.copy()

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = Quaternion(val=val, do_normalise=True)

    @property
    def dofs(self):
        return self._dofs.copy()

    @dofs.setter
    def dofs(self, val):
        self._dofs = val

    @property
    def ndofs(self):
        return self._ndofs.copy()

    @ndofs.setter
    def ndofs(self, val):
        self._ndofs = val

    @property
    def p_cam(self):
        return self._p_cam.copy()

    @p_cam.setter
    def p_cam(self, val):
        self._p_cam = val

    @property
    def q_cam(self):
        return self._q_cam

    @q_cam.setter
    def q_cam(self, val):
        self._q_cam = Quaternion(val=val, do_normalise=True)

class ErrorStates(object):
    def __init__(self, vec):
        self.size = 24
        self.set(vec)

    def set(self, vec):
        assert(len(vec) == self.size)
        self.vec = vec
        self.size = len(vec)

        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]
        dofs = vec[9:15]
        ndofs = vec[15:18]
        p_c = vec[18:21]
        theta_c = vec[21:24]

        self.dp = np.asarray(p)
        self.dv = np.asarray(v)

        dq_xyzw = quaternion_about_axis(np.linalg.norm(theta), theta)
        self.dq = Quaternion(val=dq_xyzw, do_normalise=True)

        self.ddofs = np.asarray(dofs)
        self.dndofs = np.asarray(ndofs)
        self.dpc = np.asarray(p_c)

        dqc_xyzw = quaternion_about_axis(np.linalg.norm(theta_c), theta)
        self.dqc = Quaternion(val=dqc_xyzw, do_normalise=True)

        self.theta = np.asarray(theta)
        self.theta_c = np.asarray(theta_c)

    def reset(self):
        self.set([0] * self.size)

def quaternion_about_axis(angle, axis):
    """ https://github.com/aipiano/ESEKF_IMU/blob/master/transformations.py """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = np.linalg.norm(q)

    _EPS = np.finfo(float).eps * 4.0
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)

    return np.array([*q[1:4], q[0]])