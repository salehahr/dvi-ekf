import os
import unittest

from Models import SimpleProbe, RigidSimpleProbe
from Models import Camera, Imu, n_dofs

from Filter import Quaternion

import numpy as np
import sympy as sp
t = sp.Symbol('t')

from roboticstoolbox.backends.PyPlot import PyPlot

cam = Camera(filepath='./trajs/offline_mandala0_gt.txt', max_vals=5)

def calc_R_BW(R_BC):
    """ From forward kinematics and camera data. """
    return [R_BC @ R_WC.T for R_WC in cam.R]

def do_plot(robot, q, elev=None, azim=None):
    env = PyPlot()
    env.launch()
    ax = env.fig.axes[0]
    # elev = None if elev else -90
    # azim = azim if azim else -90
    ax.view_init(elev=elev, azim=azim)
    env.add(robot)
    robot.q = q
    env.hold()

def view_selector(robot, q):
    import time
    env = PyPlot()
    env.launch()
    env.add(robot)

    ax = env.fig.axes[0]
    # xy (90, 270)
    elev, azim = 30, -60

    do_switch = False
    params = ['elev', 'azim']
    param = params[0]

    try:
        while True:
            print(f"Current elev: {elev}, current azim: {azim}.")
            ax.view_init(elev=elev, azim=azim)

            robot.q = q
            env.step()

            ans = input(f"Next {param}?")
            try:
                elev = int(ans) if (param == 'elev') else elev
                azim = int(ans) if (param == 'azim') else azim
            except ValueError:
                if ans:
                    do_switch = True

            if do_switch:
                param = [p for p in params if param != p][0]
                do_switch = False
    except:
        sys.exit()

class TestSymbolic(unittest.TestCase):
    """ Module to troubleshoot symbolic stuff. """

    @classmethod
    def setUpClass(cls):
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.acc_CB = -cls.probe.acc
        cls.R_s = sp.MatrixSymbol('R_BW', 3, 3)
        cls.acc_C_s = sp.MatrixSymbol('acc_C', 3, 1)

        cls.q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
        cls.q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
        cls.q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]

        cls.params = [*cls.q_s, *cls.q_dot_s, *cls.q_ddot_s]

    def full_expr(self):
        """ Expression for acc_B in IMU coordinates. """
        res = self.R_s @ (
                    self.acc_C_s + self.acc_CB
                    )
        return res[0][0]

    def gen_full_expr(self):
        return self.full_expr()

    def gen_partial_expr(self):
        q1dd = sp.symbols('q1_ddot')
        q5dd = sp.symbols('q5_ddot')
        q9 = sp.symbols('q9')
        # acc_C = sp.MatrixSymbol('acc_C', 3, 1)
        # R_BW = sp.MatrixSymbol('R_BW', 3, 3)

        return -q1dd*sp.sqrt(3)*sp.sin(q9) \
                + 1.0*q5dd \
                # + acc_C[0, 0]*R_BW[0, 0]

    def custom_sin(self, arg):
        return sp.sin(arg)

    def test_partial_expr(self):
        partial_expr = self.gen_partial_expr()
        func = sp.lambdify(self.params, partial_expr,
                # modules=[{'sin': custom_sin}, 'math'],
                # modules='math',
                modules='sympy',
                )

        res = func(*self.q_s, *self.q_dot_s, *self.q_ddot_s)
        print(res)

    def test_full_expr(self):
        full_expr = self.gen_partial_expr()
        print(full_expr)
        func = sp.lambdify(self.params, full_expr,
                # modules=[{'sin': custom_sin}, 'math'],
                # modules='math',
                modules='sympy',
                )

        res = func(*self.q_s, *self.q_dot_s, *self.q_ddot_s)
        print(res)

class TestSimpleProbeBC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # probe and joint variables.
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_sym]

        cls.p_BC, cls.v_BC, cls.acc_BC, cls.om_BC, alp_BC = cls.probe.get_reversed_kin()
        cls.R_BW = calc_R_BW(cls.probe.R)

    @unittest.skip("Skip plot.")
    def test_plot(self):
        view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

    def test_lambdify_R(self):
        """ Ensures correct substitutions of D.O.F.s in the expression
            for R. """
        q7 = sp.Symbol('q7')
        R_func = sp.lambdify(q7, self.probe.R, 'numpy')

class TestRigidSimpleProbe(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initialise probe and joint variables. """
        cls.probe = RigidSimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_sym]
        cls.qd = [sp.diff(q, t) for q in cls.q_0]
        cls.qdd = [sp.diff(q, t) for q in cls.qd]
        cls.joint_dofs = [*cls.q_0, *cls.qd, *cls.qdd]

        # parameters from fwkin
        cls.R_BC = cls.probe.R
        cls.p_BC, cls.v_BC, cls.acc_BC, cls.om_BC, cls.alp_BC = cls.probe.get_reversed_kin()

    @unittest.skip("Skip plot.")
    def test_plot(self):
        # view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

class TestImu(TestRigidSimpleProbe):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.filepath = './trajs/imu_test.txt'
        cls.R_BW = calc_R_BW(cls.probe.R)

        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)

    def setUp(self):
        self.imu = Imu(self.probe, cam)

    def _get_numlines(self, filepath):
        try:
            with open(filepath) as f:
                n = sum(1 for line in f)
            return n
        except FileNotFoundError:
            return 0

    def test_append_array(self):
        assert(self.imu._om == [])

        for i in range(2):
            self.imu.eval_expr_single(cam.t[i], cam.acc[:,i], cam.R[i], cam.om[:,i], cam.alp[:,i], *self.joint_dofs, append_array=True)

        assert(self.imu.om.shape == (3, 2))

    def test_write_array_to_file(self):
        self.test_append_array()
        self.imu.write_array_to_file(self.filepath)

        num_lines = self._get_numlines(self.filepath)
        self.assertEqual(num_lines, 2)

    def test_append_file(self):
        num_lines = self._get_numlines(self.filepath)

        num_data = 3
        for i in range(num_data):
            self.imu.eval_expr_single(cam.t[i], cam.acc[:,i], cam.R[i], cam.om[:,i], cam.alp[:,i], *self.joint_dofs, append_array=False, filepath=self.filepath)

        new_num_lines = self._get_numlines(self.filepath)
        self.assertEqual(new_num_lines, num_lines + num_data)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.filepath):
            os.remove(cls.filepath)
        super().tearDownClass()

class TestQuaternions(unittest.TestCase):
    def setUp(self):
        self.q1 = Quaternion(x=0, y=-0.002, z=-0.001, w=1)
        self.q2 = Quaternion(x=0, y=0, z=0, w=1)

def suite():
    suite = unittest.TestSuite()
    # suite.addTest(TestSimpleProbeBC(''))
    return suite

if __name__ == '__main__':
    # run all tests
    unittest.main(verbosity=2)

    # run only certain tests
    runner = unittest.TextTestRunner()
    runner.run(suite())