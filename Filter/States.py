from __future__ import annotations

import math
from typing import List, Optional, Union

import numpy as np

from .Quaternion import Quaternion

Rotation = Union[Quaternion, np.ndarray]


class States(object):
    def __init__(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: Rotation,
        dofs: np.ndarray,
        notch_dofs: np.ndarray,
        p_cam: np.ndarray,
        q_cam: Rotation,
    ):
        """
        :param p: W_p_BW, IMU position in world coordinates
        :param v: WW_v_BW, IMU velocity in world coordinates, differentiated w.r.t. W
        :param q: R_WB, IMU orientation w.r.t. world coordinates
        :param dofs: IMU degrees of freedom
        :param notch_dofs: notch degrees of freedom
        :param p_cam: W_p_CW, camera position in world coordinates
        :param q_cam: camera orientation w.r.t. world coordinates
        """

        # fmt: off
        self._p = np.copy(p).reshape(3,)
        self._v = np.copy(v).reshape(3,)
        self._q = Quaternion(val=q, do_normalise=True)
        self._dofs = np.copy(dofs).reshape(6,)
        self._notch_dofs = np.copy(notch_dofs).reshape(3, )
        self._p_cam = np.copy(p_cam).reshape(3,)
        self._q_cam = Quaternion(val=q_cam, do_normalise=True)
        # fmt: on

        self.size = (
            len(p)
            + len(v)
            + len(self.q.xyzw)
            + len(dofs)
            + len(notch_dofs)
            + len(p_cam)
            + len(self.q_cam.xyzw)
        )
        assert self.size == 26

        self._frozen_dofs: List[bool] = [False] * 6

    def apply_correction(self, err: ErrorStates):
        # fmt: off
        self.p += err.dp.reshape(3,)
        self.v += err.dv.reshape(3,)
        self.q = self.q * err.dq
        self.q.normalise()

        for i, fr in enumerate(self._frozen_dofs):
            if not fr:
                self._dofs[i] += err.ddofs[i]

        self.notch_dofs += err.dndofs.reshape(3, )
        self.p_cam += err.dpc.reshape(3,)
        self.q_cam = self.q_cam * err.dqc
        self.q_cam.normalise()
        # fmt: on

    def set(self, vec: List[np.ndarray]):
        self.p = vec[0].squeeze()
        self.v = vec[1].squeeze()
        self.q = vec[2].squeeze()

        for i, fr in enumerate(self._frozen_dofs):
            if not fr:
                self._dofs[i] = vec[3].squeeze()[i]

        self.notch_dofs = np.array(vec[4:7]).squeeze()
        self.p_cam = vec[7].squeeze()
        self.q_cam = vec[8].squeeze()

    def copy(self) -> States:
        """Clones current states to a new object."""
        return States(
            p=self.p,
            v=self.v,
            q=self.q,
            dofs=self.dofs,
            notch_dofs=self.notch_dofs,
            p_cam=self.p_cam,
            q_cam=self.q_cam,
        )

    def __repr__(self):
        return f"State: p_cam ({self._p_cam}), ..."

    @property
    def vec(self) -> List[np.ndarray]:
        return [
            self.p,
            self.v,
            self.q.rot,
            self.dofs,
            self.notch_dofs,
            self.p_cam,
            self.q_cam.rot,
        ]

    @property
    def p(self) -> np.ndarray:
        return self._p.copy()

    @p.setter
    def p(self, val):
        self._p = val

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @v.setter
    def v(self, val):
        self._v = val

    @property
    def q(self) -> Quaternion:
        return Quaternion(val=self._q)

    @q.setter
    def q(self, val):
        self._q = Quaternion(val=val, do_normalise=True)

    @property
    def dofs(self) -> np.ndarray:
        return self._dofs.copy()

    @dofs.setter
    def dofs(self, val):
        self._dofs = val

    @property
    def notch_dofs(self) -> np.ndarray:
        return self._notch_dofs.copy()

    @notch_dofs.setter
    def notch_dofs(self, val):
        self._notch_dofs = val

    @property
    def p_cam(self) -> np.ndarray:
        return self._p_cam.copy()

    @p_cam.setter
    def p_cam(self, val):
        self._p_cam = val

    @property
    def q_cam(self) -> Quaternion:
        return Quaternion(val=self._q_cam)

    @q_cam.setter
    def q_cam(self, val):
        self._q_cam = Quaternion(val=val, do_normalise=True)

    @property
    def frozen_dofs(self) -> List[bool]:
        return self._frozen_dofs

    @frozen_dofs.setter
    def frozen_dofs(self, value: List[bool]) -> None:
        self._frozen_dofs = value


class ErrorStates(object):
    def __init__(self, vec):
        self.size = 24
        self.set(vec)

    def set(self, vec):
        assert len(vec) == self.size
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
    """https://github.com/aipiano/ESEKF_IMU/blob/master/transformations.py"""
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = np.linalg.norm(q)

    _EPS = np.finfo(float).eps * 4.0
    if qlen > _EPS:
        q *= math.sin(angle / 2.0) / qlen
    q[0] = math.cos(angle / 2.0)

    return np.array([*q[1:4], q[0]])


def process_data(t: float, state: States) -> List[float]:
    """Appends new measurement from current state."""
    dof_rots_deg = np.rad2deg(state.dofs[:3])
    dof_trans = state.dofs[3:]
    return [
        t,
        *state.p,
        *state.v,
        *state.q.euler_xyz_deg,
        *state.q.wxyz,
        *dof_rots_deg,
        *dof_trans,
        *state.p_cam,
        *state.q_cam.euler_xyz_deg,
        *state.q_cam.wxyz,
    ]
