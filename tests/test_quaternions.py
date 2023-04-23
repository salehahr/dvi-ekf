import unittest

from dvi_ekf.tools import Quaternion


class TestQuaternions(unittest.TestCase):
    def setUp(self):
        self.q1 = Quaternion(x=0, y=-0.002, z=-0.001, w=1)
        self.q2 = Quaternion(x=0, y=0, z=0, w=1)


if __name__ == "__main__":
    from functions import run_only

    run_only(TestQuaternions)
