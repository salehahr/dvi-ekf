import numpy as np
from spatialmath import SE3
from spatialmath.base import trinterp


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
