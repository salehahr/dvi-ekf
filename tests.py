import os
import unittest

from Models import SimpleProbe, RigidSimpleProbe
from Models import Camera, Imu, n_dofs, dofs_s, dofs_cas

from Filter import Quaternion

import numpy as np
import sympy as sp
t = sp.Symbol('t')

from aux_symbolic import sympy2casadi
from casadi import *

from roboticstoolbox.backends.PyPlot import PyPlot

cam = Camera(filepath='./trajs/offline_mandala0_gt.txt', max_vals=5)

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
        # cls.acc_CB = -cls.probe.acc
        cls.R_s = sp.MatrixSymbol('R_BW', 3, 3)
        cls.acc_C_s = sp.MatrixSymbol('acc_C', 3, 1)

        cls.q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
        cls.q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
        cls.q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]

        cls.params = [*cls.q_s, *cls.q_dot_s, *cls.q_ddot_s]

    def gen_partial_expr(self):
        q1dd = sp.symbols('q1_ddot')
        q5dd = sp.symbols('q5_ddot')
        q9 = sp.symbols('q9')

        return -q1dd*sp.sqrt(3)*sp.sin(q9) \
                + 1.0*q5dd

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

    def test_tensor_mult(self):
        from sympy.abc import a, b, c, d, e, f, g, h, i, j, k, l, m, n, o
        from sympy.tensor.array import tensorproduct
        from sympy.tensor.array import tensorcontraction

        H = sp.MutableDenseNDimArray([0]*(2*3*4), (2,3,4))
        H[0,:,:] = sp.Matrix([[a, b, c, d], [e, f, g, h], [m, n, c, d]])
        H[1,:,:] = sp.Matrix([[e, f, g, h], [a, b, c, d], [e, f, g, h]])

        qd1 = sp.MutableDenseNDimArray([i, j, k, l], (4,1))

        tp1 = tensorproduct(H, qd1)
        assert(tp1.shape == (2, 3, 4, 4, 1))
        tp1 = tensorcontraction(tp1, (2,3))
        assert(tp1.shape == (2, 3, 1))
        tp1 = tp1[:,:,0]
        assert(tp1.shape == (2, 3))

        qd2 = sp.MutableDenseNDimArray([m, n, o], (3,1))
        tp2 = tensorproduct(tp1, qd2)
        assert(tp2.shape == (2, 3, 3, 1))
        tp2 = tensorcontraction(tp2, (1,2))
        assert(tp2.shape == (2, 1))

    def test_expr_with_acc(self):
        expr = self.probe.acc
        expr = self.probe.alp

class TestCasadi(unittest.TestCase):
    def setUp(self):
        """ Initialise probe and joint variables. """
        self.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

    def test_sympy2casadi(self):
        x,y = sp.symbols("x y")
        xy = sp.Matrix([x,y])
        e = sp.Matrix([x*sp.sqrt(y),sp.sin(x+y),abs(x-y)])

        X = SX.sym("xc")
        Y = SX.sym("yc")
        XY = casadi.vertcat(X,Y)

        res = sympy2casadi(e, [x, y], XY)
        print(res)

    def test_probe2casadi(self):
        T = self.probe.R
        T = self.probe.p
        J = self.probe.jacob0(self.probe.q_s)
        T = J @ dofs_s[n_dofs:2*n_dofs]
        T = self.probe._calc_acceleration()
        T = sympy2casadi(T, dofs_s, dofs_cas)

class TestSimpleProbeBC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # probe and joint variables.
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        cls.q_0 = [q if not isinstance(q, sp.Expr) else 0. for q in cls.probe.q_s]

    @unittest.skip("Skip plot.")
    def test_plot(self):
        view_selector(self.probe, self.q_0)
        do_plot(self.probe, self.q_0)

    @unittest.skip("Not necessary after implementing CasADi.")
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

class TestImu(TestRigidSimpleProbe):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.filepath = './trajs/imu_test.txt'

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
            self.imu.eval_expr_single(cam.t[i], self.probe.q_cas,
                self.probe.qd_cas, self.probe.qdd_cas,
                cam.acc[:,i], cam.R[i],
                cam.om[:,i], cam.alp[:,i], append_array=True)

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
            self.imu.eval_expr_single(cam.t[i], self.probe.q_cas,
                self.probe.qd_cas, self.probe.qdd_cas,
                cam.acc[:,i], cam.R[i], cam.om[:,i],
                cam.alp[:,i], append_array=False, filepath=self.filepath)

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

class TestFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dt = casadi.SX.sym('dt')

        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        print(cls.probe)
        print(f'q: {cls.probe.q}\n')

        # camera and imu sensors
        # num_imu_between_frames = 1
        # cam_interp = cam.interpolate(num_imu_between_frames)
        min_t, max_t = cam.t[0], cam.t[-1]

        # imu
        imu = Imu(cls.probe, cam)
        imu.eval_init(cls.probe.q, cls.probe.qd, cls.probe.qdd)
        cls.imu = imu

        # fwkin
        cls.p_CB, cls.v_CB, cls.acc_CB = cls.probe.p, cls.probe.v, cls.probe.acc
        cls.R_BC, cls.om_CB, cls.alp_CB = cls.probe.R, cls.probe.om, cls.probe.alp

        cls.states()
        cls.error_states()
        cls.inputs()
        cls.noise()

    @classmethod
    def states(cls):
        p_B = casadi.SX.sym('p_B', 3)
        v_B = casadi.SX.sym('v_B', 3)
        R_WB = casadi.SX.sym('R_WB', 3, 3)
        dofs = casadi.SX.sym('q', 6)
        dofs_tr, dofs_rot = casadi.vertsplit(dofs, [0, 3, 6])
        p_C = casadi.SX.sym('p_C', 6)

        cls.x = [p_B, v_B, R_WB, dofs_tr, dofs_rot, p_C]
        cls.x_str = ['p_B', 'v_B', 'R_WB', 'dofs_tr', 'dofs_rot', 'p_C']

        for i, x in enumerate(cls.x):
            globals()[cls.x_str[i]] = x

    @classmethod
    def error_states(cls):
        err_p_B = casadi.SX.sym('err_p_B', 3)
        err_v_B = casadi.SX.sym('err_v_B', 3)
        err_theta = casadi.SX.sym('err_theta', 3)
        err_dofs_t = casadi.SX.sym('err_dofs_t', 3)
        err_dofs_r = casadi.SX.sym('err_dofs_r', 3)
        err_p_C = casadi.SX.sym('err_p_C', 3)

        cls.err_x = [err_p_B, err_v_B, err_theta,
                    err_dofs_t, err_dofs_r, err_p_C]
        cls.err_x_str = ['err_p_B', 'err_v_B', 'err_theta',
                    'err_dofs_t', 'err_dofs_r', 'err_p_C']

        for i, err_x in enumerate(cls.err_x):
            globals()[cls.err_x_str[i]] = err_x

    @classmethod
    def inputs(cls):
        acc = casadi.SX.sym('acc', 3)
        om = casadi.SX.sym('om', 3)

        cls.u = [om, acc]
        cls.u_str = ['om', 'acc']

        for i, u in enumerate(cls.u):
            globals()[cls.u_str[i]] = u

    @classmethod
    def noise(cls):
        n_v = casadi.SX.sym('n_v', 3)
        n_om = casadi.SX.sym('n_om', 3)

        cls.n = [n_v, n_om]
        cls.n_str = ['n_v', 'n_om']

        for i, n in enumerate(cls.n):
            globals()[cls.n_str[i]] = n

    def test_fun_nominal(self):
        p_B_next = p_B + self.dt * v_B + self.dt**2 / 2 * R_WB @ acc

        fun_nominal = casadi.Function('f_nom',
            [self.dt, *self.x, *self.u],
            [   p_B_next,
                v_B + self.dt * R_WB @ acc,
                R_WB + R_WB @ casadi.skew(self.dt * om),
                dofs_t,
                dofs_r,
                p_B_next + R_WB @ self.p_CB ],
            ['dt', *self.x_str, *self.u_str],
            ['p_B_next', 'v_B_next', 'R_WB_next',
                'dofs_t_next', 'dofs_r_next', 'p_C_next'])

        res = fun_nominal(  dt  = 0.1,
                            p_B = casadi.DM([1.2, 3.9, 2.]),
                            v_B = casadi.DM([0.01, 0.02, 0.003]),
                            R_WB = casadi.DM.eye(3),
                            om = casadi.DM(self.imu.om),
                            acc = casadi.DM(self.imu.acc),
                         )
        p_B_next = res['p_B_next']

def suite():
    suite = unittest.TestSuite()
    test_class = TestSimpleProbeBC
    for t in test_class.__dict__.keys():
        if t.startswith('test'):
            suite.addTest(test_class(t))
    return suite

if __name__ == '__main__':
    # run all tests
    unittest.main(verbosity=2)

    # run only certain tests
    runner = unittest.TextTestRunner()
    runner.run(suite())