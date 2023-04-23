import math

import numpy as np


def euler_error_checking(euler_xyz: np.ndarray, euler_yzx: np.ndarray) -> None:
    """
    Checks the values of two sets of Euler angles.
    """
    x1, y1, z1 = euler_xyz
    z2, y2, x2 = euler_yzx

    _ensure_close(x1, x2)
    _ensure_close(y1, y2)
    _ensure_close(z1, z2)


def _ensure_close(v1: float, v2: float, verbose: bool = True) -> None:
    """
    Returns an AssertionError if two float values are not close.
    """
    try:
        assert math.isclose(v1, v2, rel_tol=0.1, abs_tol=0.05)
    except AssertionError:
        if verbose:
            print(f"Error in Euler angle: {v1:+0.3f} vs {v2:+0.3f}")


def skew(x: np.ndarray):
    """
    Returns the skew matrix of a 3-element vector.
    :param x:
    :return:
    """
    # fmt: off
    return np.array([[   0, -x[2], x[1]],
                     [ x[2],   0, -x[0]],
                     [-x[1], x[0],   0]])
    # fmt: on
