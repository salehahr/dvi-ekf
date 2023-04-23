from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dvi_ekf.tools.Quaternion import Quaternion


@dataclass(frozen=True)
class State(object):
    SIZE = 26

    # W_p_BW, IMU position in world coordinates
    p: np.ndarray
    # WW_v_BW, IMU velocity in world coordinates, differentiated w.r.t. W
    v: np.ndarray
    # R_WB, IMU orientation w.r.t. world coordinates
    q: Quaternion

    # IMU degrees of freedom
    dofs: np.ndarray
    # notch degrees of freedom
    notch_dofs: np.ndarray
    # W_p_CW, camera position in world coordinates
    p_cam: np.ndarray
    # camera orientation w.r.t. world coordinates
    q_cam: Quaternion

    def __post_init__(self):
        size = (
            len(self.p)
            + len(self.v)
            + len(self.q.xyzw)
            + len(self.dofs)
            + len(self.notch_dofs)
            + len(self.p_cam)
            + len(self.q_cam.xyzw)
        )
        assert size == State.SIZE

    def __repr__(self):
        return f"State: p_cam ({self._p_cam}), ..."

    def __add__(self, err: ErrorState) -> State:
        # fmt: off
        p = self.p + err.dp.reshape(3, )
        v = self.v + err.dv.reshape(3, )
        q = self.q * err.dq
        q.normalise()

        dofs = self.dofs + err.ddofs
        notch_dofs = self.notch_dofs + err.dndofs.reshape(3, )
        p_cam = self.p_cam + err.dpc.reshape(3, )
        q_cam = self.q_cam * err.dqc
        q_cam.normalise()
        # fmt: on

        return State(p, v, q, dofs, notch_dofs, p_cam, q_cam)

    @staticmethod
    def from_array(vec: List[np.ndarray]) -> State:
        """Sets values to the given ones."""
        p = vec[0].squeeze()
        v = vec[1].squeeze()
        q = Quaternion(val=vec[2].squeeze(), do_normalise=True)

        dofs = vec[3].squeeze()
        notch_dofs = np.array(vec[4:7]).squeeze()
        p_cam = vec[7].squeeze()
        q_cam = Quaternion(val=vec[8].squeeze(), do_normalise=True)

        return State(p, v, q, dofs, notch_dofs, p_cam, q_cam)

    @property
    def array(self) -> List[np.ndarray]:
        return [
            self.p,
            self.v,
            self.q.rot,
            self.dofs,
            self.notch_dofs,
            self.p_cam,
            self.q_cam.rot,
        ]


@dataclass(frozen=True)
class ErrorState(object):
    SIZE = State.SIZE - 2

    dp: np.ndarray
    dv: np.ndarray
    dq: Quaternion

    ddofs: np.ndarray
    dndofs: np.ndarray
    dpc: np.ndarray
    dqc: Quaternion

    theta: np.ndarray
    theta_c: np.ndarray

    @staticmethod
    def from_array(vec: np.ndarray) -> ErrorState:
        assert len(vec) == ErrorState.SIZE

        p = vec[0:3]
        v = vec[3:6]
        theta = vec[6:9]
        dofs = vec[9:15]
        ndofs = vec[15:18]
        p_c = vec[18:21]
        theta_c = vec[21:24]

        dp = np.asarray(p)
        dv = np.asarray(v)
        dq = Quaternion.about_axis(np.linalg.norm(theta), axis=theta)

        ddofs = np.asarray(dofs)
        dndofs = np.asarray(ndofs)
        dpc = np.asarray(p_c)
        dqc = Quaternion.about_axis(np.linalg.norm(theta_c), axis=theta)

        theta = np.asarray(theta)
        theta_c = np.asarray(theta_c)

        return ErrorState(dp, dv, dq, ddofs, dndofs, dpc, dqc, theta, theta_c)
