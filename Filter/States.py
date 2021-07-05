import numpy as np
from .Quaternion import Quaternion

class States(object):
    def __init__(self, p, v, q, dofs, p_cam):
        self._p = np.asarray(p).reshape(3,1)
        self._v = np.asarray(v).reshape(3,1)
        self._q = None
        self._dofs = dofs
        self._p_cam = p_cam

        self.q = q

        self.size = len(p) + len(v) + len(self.q.xyzw) \
                    + len(dofs) + len(p_cam)
        assert(self.size == 19)

    def apply_correction(self, err):
        self.p += err.dp.reshape(3,1)
        self.v += err.dv.reshape(3,1)
        self.q = self.q * err.dq
        self.q.normalise()

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
        return Quaternion(val=self._q, do_normalise=True)

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
    def p_cam(self):
        return self._p_cam.copy()

    @p_cam.setter
    def p_cam(self, val):
        self._p_cam = val

class ErrorStates(object):
    def __init__(self, vec):
        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]

        self.dp = np.asarray(p)
        self.dv = np.asarray(v)
        self.dq = Quaternion(v=theta/2, w=1.)
