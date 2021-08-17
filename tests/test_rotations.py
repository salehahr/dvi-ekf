import unittest

import casadi
import numpy as np

from spatialmath import SE3

def np_string(arr):
    return np.array2string(arr,
                    precision=2,
                    suppress_small=True)

def compare(orig, result, answer):
    np.testing.assert_allclose(result, np.array(answer), atol=0.001)
    print(f'\nRotated \n{orig} to \n{np_string(result)}.')

class TestRotations(unittest.TestCase):
    def setUp(self):
        self.p0 = p = [1, 1, 1]
        self.R0 = np.eye(3)
        self.rot_ang_in_rad = 45 * np.pi / 180

        # reference values
        self.p_active = [1, 0, 1.414]
        self.R_active = np.array([[1,     0,      0],
                                  [0, 0.707, -0.707],
                                  [0, 0.707,  0.707]])

    def test_active_rot(self):
        R_x = SE3.Rx(self.rot_ang_in_rad).R

        # rotation of a point
        # point is rotated 45 deg clockwise in current frame
        p        = R_x @ self.p0
        compare(self.p0, p, self.p_active)

        # rotation of a frame
        # frame is rotated 45 clockwise
        R        = R_x @ self.R0
        compare(self.R0, R, self.R_active)

    def test_passive_rot(self):
        R_x = SE3.Rx(self.rot_ang_in_rad).R

        # rotation of a point
        # point is rotated 45 deg clockwise in current frame
        p        = R_x.T @ self.p0
        p_passive = [1, 1.414, 0]
        compare(self.p0, p, p_passive)

        # rotation of a frame
        # old frame is -45 deg relative to new frame
        R = R_x.T @ self.R0
        R_passive = self.R_active.T
        compare(self.R0, R, R_passive)

if __name__ == '__main__':
    from functions import run_only
    run_only(TestRotations)