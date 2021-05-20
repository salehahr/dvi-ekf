import unittest

from Filter import Quaternion

class TestQuats(unittest.TestCase):
    def test_quat_multiplication(self):
        quat = Quaternion(xyzw=[0.1, 0.1, 0.2, 0.1], do_normalise=True)
        print(quat)

if __name__ == '__main__':
    unittest.main()