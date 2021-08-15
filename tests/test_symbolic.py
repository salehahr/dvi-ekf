import unittest

import sympy as sp
import casadi

from context import SimpleProbe
from context import syms, symfcns

class TestSymbolic(unittest.TestCase):
    """ Module to troubleshoot symbolic stuff. """

    @classmethod
    def setUpClass(cls):
        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)
        # cls.acc_CB = -cls.probe.acc
        cls.R_s = sp.MatrixSymbol('R_BW', 3, 3)
        cls.acc_C_s = sp.MatrixSymbol('acc_C', 3, 1)

        cls.q_s = [sp.Symbol(f'q{x}') for x in range(1,syms.num_dofs+1)]
        cls.q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,syms.num_dofs+1)]
        cls.q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,syms.num_dofs+1)]

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

        X = casadi.SX.sym("xc")
        Y = casadi.SX.sym("yc")
        XY = casadi.vertcat(X,Y)

        res = symfcns.sympy2casadi(e, [x, y], XY)
        print(res)

    def test_probe2casadi(self):
        T = self.probe.R
        T = self.probe.p
        J = self.probe.jacob0(self.probe.q_s)
        T = J @ syms.dofs_s[syms.num_dofs:2*syms.num_dofs]
        T = self.probe._calc_acceleration()
        T = symfcns.sympy2casadi(T, syms.dofs_s, syms.dofs_cas)

if __name__ == '__main__':
    from functions import run_only
    run_only(TestSymbolic)
    run_only(TestCasadi)