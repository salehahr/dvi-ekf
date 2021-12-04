from test_camera import TestCamera
from test_filter import TestFilter
from test_imu import TestImu
from test_quaternions import TestQuaternions
from test_rigidsimpleprobe import TestRigidSimpleProbe
from test_simpleprobe import TestSimpleProbe
from test_symbolic import TestCasadi, TestSymbolic

if __name__ == "__main__":
    from functions import run_all

    run_all()
