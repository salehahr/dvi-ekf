import os
import unittest

from context import Imu
from data import camera
from test_rigidsimpleprobe import TestRigidSimpleProbe


class TestImu(TestRigidSimpleProbe):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.filepath = "./trajs/imu_test.txt"

        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)

    def setUp(self):
        self.imu = Imu(self.probe, camera)

    def _get_numlines(self, filepath):
        try:
            with open(filepath) as f:
                n = sum(1 for line in f)
            return n
        except FileNotFoundError:
            return 0

    def test_append_array(self):
        assert self.imu._om == []

        for i in range(2):
            self.imu.eval_expr_single(
                camera.t[i],
                self.probe.q_cas,
                self.probe.qd_cas,
                self.probe.qdd_cas,
                camera.acc[:, i],
                camera.R[i],
                camera.om[:, i],
                camera.alp[:, i],
                append_array=True,
            )

        assert self.imu.om.shape == (3, 2)

    def test_write_array_to_file(self):
        self.test_append_array()
        self.imu.write_array_to_file(self.filepath)

        num_lines = self._get_numlines(self.filepath)
        self.assertEqual(num_lines, 2)

    def test_append_file(self):
        num_lines = self._get_numlines(self.filepath)

        num_data = 3
        for i in range(num_data):
            self.imu.eval_expr_single(
                camera.t[i],
                self.probe.q_cas,
                self.probe.qd_cas,
                self.probe.qdd_cas,
                camera.acc[:, i],
                camera.R[i],
                camera.om[:, i],
                camera.alp[:, i],
                append_array=False,
                filepath=self.filepath,
            )

        new_num_lines = self._get_numlines(self.filepath)
        self.assertEqual(new_num_lines, num_lines + num_data)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)
        super().tearDownClass()


if __name__ == "__main__":
    from functions import run_only

    run_only(TestImu)
