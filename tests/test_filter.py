import unittest

import casadi
import numpy as np

from context import SimpleProbe, Camera, Imu
from context import syms

from data import camera

class TestFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dt = casadi.SX.sym('dt')

        cls.probe = SimpleProbe(scope_length=0.5, theta_cam=np.pi/6)
        print(cls.probe)
        print(f'q: {cls.probe.q}\n')

        # camera and imu sensors
        # num_imu_between_frames = 1
        # cam_interp = camera.interpolate(num_imu_between_frames)
        min_t, max_t = camera.t[0], camera.t[-1]

        # imu
        imu = Imu(cls.probe, camera)
        imu.eval_init()
        cls.imu = imu

        # fwkin
        cls.p_CB, cls.v_CB, cls.acc_CB = cls.probe.p, cls.probe.v, cls.probe.acc
        cls.R_BC, cls.om_CB, cls.alp_CB = cls.probe.R, cls.probe.om, cls.probe.alp

        cls.states()
        cls.error_states()
        cls.measurements()
        cls.inputs()
        cls.noise()

    @classmethod
    def states(cls):
        cls.p_B = casadi.SX.sym('p_B', 3)
        cls.v_B = casadi.SX.sym('v_B', 3)
        cls.R_WB = casadi.SX.sym('R_WB', 3, 3)

         # note: creating cls.dofs via casadi.SX.sym results in free variables in the functions created later on -- therefore syms.dofs_cas, which was used in Probe.py, has to be imported
        cls.dofs, cls.ddofs, cls.dddofs = casadi.vertsplit(syms.dofs_cas, [0, 8, 16, 24])
        cls.dofs_t, cls.dofs_r, _ = casadi.vertsplit(cls.dofs, [0, 3, 6, 8])

        cls.p_C = casadi.SX.sym('p_C', 6)

        cls.x = [cls.p_B, cls.v_B, cls.R_WB, cls.dofs_t, cls.dofs_r, cls.p_C]
        cls.x_str = ['p_B', 'v_B', 'R_WB', 'dofs_t', 'dofs_r', 'p_C']

    @classmethod
    def error_states(cls):
        cls.err_p_B = casadi.SX.sym('err_p_B', 3)
        cls.err_v_B = casadi.SX.sym('err_v_B', 3)
        cls.err_theta = casadi.SX.sym('err_theta', 3)
        cls.err_dofs_t = casadi.SX.sym('err_dofs_t', 3)
        cls.err_dofs_r = casadi.SX.sym('err_dofs_r', 3)
        cls.err_p_C = casadi.SX.sym('err_p_C', 3)

        cls.err_x = [cls.err_p_B, cls.err_v_B, cls.err_theta,
                    cls.err_dofs_t, cls.err_dofs_r, cls.err_p_C]
        cls.err_x_str = ['err_p_B', 'err_v_B', 'err_theta',
                    'err_dofs_t', 'err_dofs_r', 'err_p_C']

    @classmethod
    def measurements(cls):
        _, cls.q_notch, _ = casadi.vertsplit(cls.dofs, [0, 6, 7, 8])
        _, cls.qd_notch, _ = casadi.vertsplit(cls.ddofs, [0, 6, 7, 8])
        cls.notch_dof = [cls.q_notch, cls.qd_notch]
        cls.notch_dof_str = ['q_notch', 'qd_notch']

    @classmethod
    def inputs(cls):
        cls.acc = casadi.SX.sym('acc', 3)
        cls.om = casadi.SX.sym('om', 3)

        cls.u = [cls.om, cls.acc]
        cls.u_str = ['om', 'acc']

    @classmethod
    def noise(cls):
        cls.n_v = casadi.SX.sym('n_v', 3)
        cls.n_om = casadi.SX.sym('n_om', 3)
        cls.n_dofs_t = casadi.SX.sym('n_dofs_t', 3)
        cls.n_dofs_r = casadi.SX.sym('n_dofs_r', 3)

        cls.n = [cls.n_v, cls.n_om, cls.n_dofs_t, cls.n_dofs_r]
        cls.n_str = ['n_v', 'n_om', 'n_dofs_t', 'n_dofs_r']

    def test_fun_nominal(self):
        p_B_next = self.p_B \
                + self.dt * self.v_B \
                + self.dt**2 / 2 * self.R_WB @ self.acc

        fun_nominal = casadi.Function('f_nom',
            [self.dt, *self.x, *self.u, *self.notch_dof],
            [   p_B_next,
                self.v_B + self.dt * self.R_WB @ self.acc,
                self.R_WB + self.R_WB @ casadi.skew(self.dt * self.om),
                self.dofs_t,
                self.dofs_r,
                p_B_next + self.R_WB @ self.p_CB ],
            ['dt', *self.x_str, *self.u_str, *self.notch_dof_str],
            ['p_B_next', 'v_B_next', 'R_WB_next',
                'dofs_t_next', 'dofs_r_next', 'p_C_next'])

        res = fun_nominal(  dt  = 0.1,
                            p_B = casadi.DM([1.2, 3.9, 2.]),
                            v_B = casadi.DM([0.01, 0.02, 0.003]),
                            R_WB = casadi.DM.eye(3),
                            om = casadi.DM(self.imu.om),
                            acc = casadi.DM(self.imu.acc),
                            q_notch = casadi.DM(0.),
                            qd_notch = casadi.DM(0.),
                         )
        p_B_next = res['p_B_next']

    def _derive_err_pc_dot(self):
        """
            Example derivation of err_p_B:

            [In continuous time]
            p_B_tr_dot = p_B_dot + err_p_B_dot
            v_B_tr = v_B + err_v_B

            err_p_B_dot = v_B_tr - v_B
                        = v_B + err_v_B - v_B
                        = err_v_B

            [Discretised]
            err_p_B_next = err_p_B + dt * err_v_B
        """

        # deriving err_p_C_dot -- define the true values
        v_B_tr = self.v_B + self.err_v_B
        R_WB_tr = self.R_WB @ (casadi.DM.eye(3) \
                    + casadi.skew(self.err_theta))
        dofs_t_tr = self.dofs_t + self.err_dofs_t
        dofs_r_tr = self.dofs_r + self.err_dofs_r # this is a placeholder ## TODO
        dofs_tr = casadi.vertcat(dofs_t_tr, dofs_r_tr)
        om_tr = self.om - self.n_om

        # deriving err_p_C_dot -- continuous time
        p_CB_dot = self.R_WB @ self.v_CB \
                + casadi.skew(self.om) @ self.R_WB @ self.p_CB
        p_CB_dot_tr = R_WB_tr @ self.v_CB \
                + casadi.skew(om_tr) @ R_WB_tr @ self.p_CB

        p_C_dot = self.v_B + p_CB_dot
        p_C_dot_tr = v_B_tr + p_CB_dot_tr

        # err_p_C_dot = p_C_dot_tr - p_C_dot # results in free variables v_B
        err_p_C_dot = self.err_v_B + p_CB_dot_tr - p_CB_dot

        return err_p_C_dot

    def test_fun_error(self):
        err_p_C_dot = self._derive_err_pc_dot()

        fun_error = casadi.Function('f_err',
            [self.dt, *self.err_x, *self.u, *self.n, self.R_WB],
            [   self.err_p_B + self.dt * self.err_v_B,
                self.err_v_B + self.dt * (-self.R_WB @ casadi.skew(self.acc) @ self.err_theta) + self.n_v,
                -casadi.cross(self.om, self.err_theta) + self.n_om,
                self.n_dofs_t,
                self.n_dofs_r,
                self.err_p_C + self.dt * err_p_C_dot ],
            ['dt', *self.err_x_str, *self.u_str,
                *self.n_str, 'R_WB'],
            ['err_p_B_next', 'err_v_B_next', 'err_theta_next',
                'err_dofs_t_next', 'err_dofs_r_next',
                'err_p_C_next'])

        res = fun_error(  dt  = 0.1,
                        err_p_B = casadi.DM([1.2, 3.9, 2.]),
                        err_v_B = casadi.DM([0.01, 0.02, 0.003]),
                        R_WB = casadi.DM.eye(3),
                        om = casadi.DM(self.imu.om),
                        acc = casadi.DM(self.imu.acc),
                         )
        err_p_B_next = res['err_p_B_next']

        index_x = [fun_error.index_in(x) for x in self.err_x_str]
        index_n = [fun_error.index_in(n) for n in self.n_str]
        index_err_p_C = fun_error.index_out('err_p_C_next')

        jac_pc_x = []
        for idx in index_x:
            name = 'jac_pc_' + self.err_x_str[idx - index_x[0]]

            jac = fun_error.jacobian_old(idx, index_err_p_C)
            jac = jac.slice(name, range(0,jac.n_in()), [0])
            jac_pc_x.append(jac)

        jac_pc_n = []
        for idn in index_n:
            name = 'jac_pc_' + self.n_str[idn - index_n[0]]

            jac = fun_error.jacobian_old(idn, index_err_p_C)
            jac = jac.slice(name, range(0,jac.n_in()), [0])
            jac_pc_n.append(jac)

if __name__ == '__main__':
    from functions import run_only
    run_only(TestFilter)