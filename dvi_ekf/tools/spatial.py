from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spatialmath import SE3, UnitQuaternion
from spatialmath.base import trinterp
from sympy.abc import a, b, c
from sympy.matrices import Matrix
from sympy.solvers import solve

if TYPE_CHECKING:
    from dvi_ekf.tools.Quaternion import Quaternion

# fmt: off
Om = Matrix([[0, -a, -b, -c],
             [a,  0,  c, -b],
             [b, -c,  0,  a],
             [c,  b, -a,  0]])
# fmt: on


def interpolate(frames: SE3, n_interframe: int) -> SE3:
    spacer = np.linspace(start=0, stop=1, num=n_interframe + 1)
    frames_interp = frames[0]

    F_prev = SE3(frames_interp)
    for F in frames[1:]:
        for s in spacer[1:]:
            F_s = SE3(trinterp(F_prev.A, F.A, s=s))
            frames_interp.append(F_s)
        F_prev = F

    return frames_interp


def solve_for_omega(
    q0: Quaternion, q1: Quaternion, dt: float, eval_at_q0: bool = True
) -> np.ndarray:
    qdiff = (q1.wxyz - q0.wxyz) / dt

    if eval_at_q0:
        eqn = Om @ q0.wxyz - 2 * qdiff
    else:
        eqn = Om @ q1.wxyz - 2 * qdiff

    omega = np.array(list(solve(eqn[1:], [a, b, c]).values()))

    return np.array(omega).astype(np.float64)


def get_omega_local(imu_frames: SE3) -> np.ndarray:
    # initialise arrays
    imu_q = UnitQuaternion(imu_frames)
    dq = UnitQuaternion.Alloc(len(imu_frames))

    # error quaternion calculation
    for i, q in enumerate(imu_q[1:]):
        dq[i] = imu_q[i - 1].conj() * imu_q[i]
    dq[0] = UnitQuaternion()

    return np.array([np.multiply(*a.angvec()) for a in dq])
