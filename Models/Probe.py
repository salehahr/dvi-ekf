import roboticstoolbox as rtb
import sympy as sp
from casadi import *
from spatialmath import SE3
from spatialmath.base.symbolic import issymbol
from sympy.tensor.array import tensorcontraction, tensorproduct

from symbolics import symbols as syms
from symbolics.functions import to_casadi

# just so that the plot is orientated correctly...
plot_rotation = SE3.Ry(-180, "deg")


class Probe(rtb.DHRobot):
    def __init__(self, scope_length, theta_cam):
        """Initialises robot links."""

        links = self._gen_links(scope_length, theta_cam)
        super().__init__(links, name="probe", base=plot_rotation)

        self.q_s = syms.q_s.copy()
        self.qd_s = syms.qd_s.copy()
        self.qdd_s = syms.qdd_s.copy()

    @staticmethod
    def _gen_links(scope_length, theta_cam):
        """Generates robot links according to chosen configuration."""
        links_imu_to_cam = [
            # imu orientation
            rtb.RevoluteDH(alpha=sp.pi / 2, offset=sp.pi / 2),
            rtb.RevoluteDH(alpha=-sp.pi / 2, offset=-sp.pi / 2),
            rtb.RevoluteDH(alpha=0, offset=0),
            # # imu translation
            rtb.PrismaticDH(theta=0, alpha=sp.pi / 2),
            rtb.PrismaticDH(theta=sp.pi / 2, alpha=sp.pi / 2),
            rtb.PrismaticDH(theta=-sp.pi / 2, alpha=sp.pi / 2),
        ]

        links_cam_to_slam = [
            # cam to scope end
            rtb.RevoluteDH(d=scope_length, a=0, alpha=theta_cam, offset=0),
            # scope end to virtual slam
            rtb.RevoluteDH(d=0 * scope_length, a=0, alpha=0, offset=0),
        ]

        return links_imu_to_cam + links_cam_to_slam

    @property
    def q_cas(self):
        return to_casadi(self.q_s)

    @property
    def qd_cas(self):
        return to_casadi(self.qd_s)

    @property
    def qdd_cas(self):
        return to_casadi(self.qdd_s)

    @property
    def qd_s_arr(self):
        return sp.MutableDenseNDimArray(self.qd_s, (self.n, 1))

    @property
    def fwkin(self):
        return self.get_sym(self.q_s)

    @property
    def T(self):
        return self.fkine(self.q_s)

    @property
    def R(self):
        R = self.T.R
        return to_casadi(R)

    @property
    def p(self):
        p = self.T.t
        return to_casadi(p)

    # ---  velocity calculations ---
    def _calc_velocity(self):
        J = self.jacob0(self.q_s)
        return J @ self.qd_s

    @property
    def v(self):
        v = self._calc_velocity()[:3]
        return to_casadi(v)

    @property
    def om(self):
        om = self._calc_velocity()[-3:]
        return to_casadi(om)

    # --- acceleration calculations ---
    def hessian_symbolic(self, J0):
        n = self.n
        H = sp.MutableDenseNDimArray([0] * (6 * n * n), (6, n, n))

        for j in range(n):
            for i in range(j, n):
                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    # --- acceleration calculations ---
    def _calc_acceleration(self):
        J0 = self.jacob0(self.q_s)
        H = self.hessian_symbolic(J0)

        tp1 = tensorcontraction(tensorproduct(H, self.qd_s_arr), (2, 3))  # 6x8x8x8x1
        tp1 = tp1[:, :, 0]  # 6x8

        tp2 = tensorcontraction(tensorproduct(tp1, self.qd_s_arr), (1, 2))  # 6x1
        tp2 = tp2.reshape(
            6,
        )

        # H @ qd @ qd + J @ qdd
        res = tp2 + J0 @ self.qdd_s
        # res = sp.Array(tp2 + (J0 @ self.qdd_s).reshape(6,1))
        return res

    @property
    def acc(self):
        acc = self._calc_acceleration()[:3]
        return to_casadi(acc)

    @property
    def alp(self):
        alp = self._calc_acceleration()[-3:]
        return to_casadi(alp)

    @property
    def joint_dofs(self) -> list:
        return [self.q_cas, self.qd_cas, self.qdd_cas]

    @property
    def imu_dofs(self):
        return [*self.q[:6]]

    def __str__(self):
        """Modified to enable printing of table with symbolic offsets."""

        offsets = [L.offset for L in self]
        offset_sym = [issymbol(offset) for offset in offsets]

        # convert symbolic offsets to float
        for i, is_sym in enumerate(offset_sym):
            if is_sym:
                self[i].offset = float(self[i].offset)

        s = super().__str__()

        # revert to symbolic
        for i, is_sym in enumerate(offset_sym):
            if is_sym:
                self[i].offset = offsets[i]

        return s

    def get_sym(self, q):
        # set joint dofs to symbolic
        q_orig = self.q_s.copy()
        self.q_s = q.copy()

        p = self.p
        R = self.R

        v = self.v
        om = self.om

        acc = self.acc
        alp = self.alp

        # reset joint dofs
        self.q_s = q_orig

        return p, R, v, om, acc, alp

    def plot(
        self, config, block=True, limits=None, movie=None, is_static=True, dt=0.05
    ):
        """Modified to allow 'hold' of figure."""

        import tkinter

        from roboticstoolbox.backends.PyPlot import PyPlot

        print(f"Plotting robot with the configuration:\n\t {config}")

        env = PyPlot()

        # visuals
        limit_x = [-0.1, 0.1]
        limit_y = [-0.7, 0.1]
        limit_z = [0.0, 0.7]
        limits = [*limit_x, *limit_y, *limit_z] if (limits is None) else limits

        env.launch(limits=limits)
        ax = env.fig.axes[0]
        try:
            ax.view_init(azim=azim, elev=elev)
        except NameError:
            pass  # default view

        # robots
        env.add(self, jointlabels=True, jointaxes=False, eeframe=True, shadow=False)

        # save gif
        loop = True if (movie is None) else False
        images = []

        if not is_static:
            try:
                while True:
                    for qk in config:
                        self.q = qk
                        env.step(dt)

                        if movie is not None:
                            images.append(env.getframe())

                    if movie is not None:
                        # save it as an animated gif
                        images[0].save(
                            movie,
                            save_all=True,
                            append_images=images[1:],
                            optimize=False,
                            duration=dt,
                            loop=0,
                        )
                    if not loop:
                        break

                if block:
                    env.hold()

            except tkinter.TclError:
                # handles error when closing the window
                return None
        else:
            try:
                self.q = config
                env.step(dt)

                if block:
                    env.hold()
            except tkinter.TclError:
                # handles error when closing the window
                return None


class SimpleProbe(Probe):
    """Simple probe with ROT9 as the only degree of freedom."""

    ROT1, ROT2, ROT3 = 0.0, 0.0, 0.0
    TRANS4, TRANS5, TRANS6 = 0.0, 0.0, 20

    constraints = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, None, 0.0]

    def __init__(self, scope_length, theta_cam):
        """Initialises probe as normal and sets the generalised
        coordinates, which are to be constrained, to the
        corresponding constant values."""

        super().__init__(scope_length, theta_cam)

        # populate dof vectors q, qd, qdd with constraints
        constraints = self.__class__.constraints
        assert len(constraints) == len(self.q_s)

        for i, c in enumerate(constraints):
            if c is not None:
                self.q[i] = c
                self.q_s[i] = c

                self.qd[i] = 0
                self.qd_s[i] = 0

                self.qdd[i] = 0
                self.qdd_s[i] = 0


class RigidSimpleProbe(SimpleProbe):
    """Simple probe with non-rotating, non-translating parts.
    Uses the BC configuration."""

    ROT1, ROT2, ROT3 = 0.0, 0.0, 0.0
    TRANS4, TRANS5, TRANS6 = 0.0, 0.0, 20

    constraints = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, 0.0, 0.0]

    def __init__(self, scope_length, theta_cam):
        super().__init__(scope_length, theta_cam)

        # set all joint values to 0 except for the given constant values
        self.q = [q if not isinstance(q, sp.Expr) else 0.0 for q in self.q_s]
        self.qd = [0] * self.n
        self.qdd = [0] * self.n


class SymProbe(object):
    """Container class for probe that only stores the
    symbolic forward kinematics relations."""

    def __init__(self, probe, const_dofs=False):
        self.n = probe.n
        self.q0 = probe.q.copy()

        self.q = syms.q_s.copy() if not const_dofs else self.q0
        # set non-imu, non-notch dofs to zero
        self.q[-1] = 0

        self.qd = probe.qd_cas
        self.qdd = probe.qdd_cas

        # relative kinematics in terms of q0, q1, ...
        self.sym_fwkin = probe.get_sym(self.q)
        p, R, v, om, acc, alp = self.sym_fwkin
        self.p = p
        self.R = R
        self.v = v
        self.om = om
        self.acc = acc
        self.alp = alp

        self.p_tr = self._get_tr(p)
        self.R_tr = self._get_tr(R)
        self.v_tr = self._get_tr(p)
        self.om_tr = self._get_tr(om)
        self.acc_tr = self._get_tr(acc)
        self.alp_tr = self._get_tr(alp)

        self._flag_const_dofs = const_dofs

    def _get_tr(self, expr):
        """Returns kinematic expressions in terms of
        estimated DOFs and error DOFs."""
        return casadi.substitute(expr, syms.q_cas, syms.q_tr_cas)

    def get_est_fwkin(self, dofs_est, notchdofs):
        """Returns the estimated relative kinematics
        using the estimated dofs."""
        f_est_probe = casadi.Function(
            "f_est_probe",
            [syms.dofs, syms.notchdofs],
            [*self.sym_fwkin],
            ["dofs", "notchdofs"],
            [*syms.probe_fwkin_str],
        )
        return [casadi.DM(r).full() for r in f_est_probe(dofs_est, notchdofs)]
