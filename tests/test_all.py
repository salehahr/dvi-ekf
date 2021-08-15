from test_symbolic import TestSymbolic, TestCasadi
from test_camera import TestCamera
from test_simpleprobe import TestSimpleProbe
from test_rigidsimpleprobe import TestRigidSimpleProbe
from test_imu import TestImu
from test_quaternions import TestQuaternions
from test_filter import TestFilter

if __name__ == '__main__':
    from functions import run_all
    run_all()