import unittest

import numpy as np
import sympy as sp

from context import RigidSimpleProbe, SymProbe
from data import t

class TestRigidSimpleProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initialise probe and joint variables. """
        cls.probe = RigidSimpleProbe(scope_length=0.5, theta_cam=np.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_s]
        cls.qd = [sp.diff(q, t) for q in cls.q_0]
        cls.qdd = [sp.diff(q, t) for q in cls.qd]
        cls.joint_dofs = [*cls.q_0, *cls.qd, *cls.qdd]

        # parameters from fwkin
        cls.R_BC = cls.probe.R

    @unittest.skip("Skip plot.")
    def test_plot(self):
        # view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

    def test_sym_probe(self):
        sym_probe = SymProbe(self.probe)
        print(self.probe.q)
        print(sym_probe.q)
        
        print()
        print(self.probe.p)
        print(sym_probe.p)
        print(sym_probe.p_tr)

        print()
        print(self.probe.R)
        print(sym_probe.R)
        print(sym_probe.R_tr)

if __name__ == '__main__':
    from functions import run_only
    run_only(TestRigidSimpleProbe)