import unittest

import sympy as sp
from context import SimpleProbe
from functions import do_plot, view_selector


class TestSimpleProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # probe and joint variables.
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi / 6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0.0 for q in cls.probe.q_s]

    @unittest.skip("Skip plot.")
    def test_plot(self):
        view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)


if __name__ == "__main__":
    from functions import run_only

    run_only(TestSimpleProbe)
