import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath.base.symbolic as sym

import numpy as np
import sympy as sp
from sympy.tensor.array import tensorproduct, tensorcontraction

from .params import q_s, qd_s, qdd_s
from .params import q_cas, qd_cas, qdd_cas

# just so that the plot is orientated correctly...
plot_rotation = SE3.Ry(-180, 'deg')

def dummify_undefined_functions(expr):
    mapping = {}

    # replace all Derivative terms
    for der in expr.atoms(sp.Derivative):
        f_name = der.expr.func.__name__
        der_count = der.derivative_count
        ds = 'd' * der_count
        # var_names = [var.name for var in der.variables]
        # name = "d%s_d%s" % (f_name, 'd'.join(var_names))
        name = f"{f_name}_{ds}ot"# % (f_name, 'd'.join(var_names))
        mapping[der] = sp.Symbol(name)

    # replace undefined functions
    from sympy.core.function import AppliedUndef
    for f in expr.atoms(AppliedUndef):
        f_name = f.func.__name__
        mapping[f] = sp.Symbol(f_name)

    return expr.subs(mapping)

def dummify_array(expr):
    is_array = isinstance(expr, (list, np.ndarray))

    if is_array:
        for i, a in enumerate(expr):
            expr[i] = dummify_undefined_functions(a)
        return expr
    else:
        return dummify_undefined_functions(expr)

class Probe(rtb.DHRobot):
    def __init__(self, scope_length, theta_cam):
        """ Initialises robot links. """

        links = self._gen_links(scope_length, theta_cam)
        super().__init__(links, name='probe', base=plot_rotation)

        self.q_s = q_s.copy()
        self.qd_s = qd_s.copy()
        self.qdd_s = qdd_s.copy()

    def _gen_links(self, scope_length, theta_cam):
        """ Generates robot links according to chosen configuration. """
        links_imu_to_cam = [
            # imu orientation
            rtb.RevoluteDH(alpha=sp.pi/2, offset=sp.pi/2),
            rtb.RevoluteDH(alpha=-sp.pi/2, offset=-sp.pi/2),
            rtb.RevoluteDH(alpha=0, offset=0),
            # # imu translation
            rtb.PrismaticDH(theta=0, alpha=sp.pi/2),
            rtb.PrismaticDH(theta=sp.pi/2, alpha=sp.pi/2),
            rtb.PrismaticDH(theta=-sp.pi/2, alpha=sp.pi/2),
            ]

        links_cam_to_slam = [
            # cam to scope end
            rtb.RevoluteDH(d=scope_length, a=0, alpha=theta_cam, offset=0),
            # scope end to virtual slam
            rtb.RevoluteDH(d=0*scope_length, a=0, alpha=0, offset=0),
            ]

        return links_imu_to_cam + links_cam_to_slam

    @property
    def qd_s_arr(self):
        return sp.MutableDenseNDimArray(self.qd_s, (self.n, 1))

    @property
    def T(self):
        return self.fkine(self.q_s)

    @property
    def R(self):
        R = self.T.R

        # evaluate sp expressions -- might break
        for i, row in enumerate(R):
            for j, col in enumerate(row):
                R[i,j] = R[i,j].evalf()

        return R

    @property
    def p(self):
        return self.T.t.reshape(3,1)

    ## ---  velocity calculations --- ##
    def _calc_velocity(self):
        J = self.jacob0(self.q_s)
        return J @ self.qd_s

    @property
    def v(self):
        v = self._calc_velocity()[:3]
        return dummify_array(v).reshape(3,1)

    @property
    def om(self):
        om = self._calc_velocity()[-3:]
        return dummify_array(om).reshape(3,1)

    ## --- acceleration calculations --- ##
    def hessian_symbolic(self, J0):
        n = self.n
        H = sp.MutableDenseNDimArray([0]*(6*n*n), (6, n, n))

        for j in range(n):
            for i in range(j, n):
                H[:3, i, j] = np.cross(J0[3:, j], J0[:3, i])
                H[3:, i, j] = np.cross(J0[3:, j], J0[3:, i])

                if i != j:
                    H[:3, j, i] = H[:3, i, j]

        return H

    ## --- acceleration calculations --- ##
    def _calc_acceleration(self):
        J0 = self.jacob0(self.q_s)
        H = self.hessian_symbolic(J0)

        tp1 = tensorcontraction(tensorproduct(H, self.qd_s_arr), (2,3)) # 6x8x8x8x1
        tp1 = tp1[:,:,0] # 6x8

        tp2 = tensorcontraction(tensorproduct(tp1, self.qd_s_arr), (1,2)) # 6x1

        # H @ qd @ qd + J @ qdd
        return sp.Array(tp2 + (J0 @ self.qdd_s).reshape(6,1))

    @property
    def acc(self):
        a = self._calc_acceleration()[:3]
        return dummify_array(a).reshape(3,1)

    @property
    def alp(self):
        alp = self._calc_acceleration()[-3:]
        return dummify_array(alp).reshape(3,1)

    def __str__(self):
        """ Modified to enable printing of table with symbolic offsets."""

        offsets = [L.offset for L in self]
        offset_sym = [sym.issymbol(offset) for offset in offsets]

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

    def plot(self, config, block=True, limits=None, movie=None, is_static=True, dt=0.05):
        """ Modified to allow 'hold' of figure. """

        from roboticstoolbox.backends.PyPlot import PyPlot
        import tkinter
        print(f"Plotting robot with the configuration:\n\t {config}")

        env = PyPlot()

        # visuals
        limit_x = [-0.1, 0.1]
        limit_y = [-0.7, 0.1]
        limit_z = [0., 0.7]
        limits = [*limit_x, *limit_y, *limit_z] if (limits is None) else limits

        env.launch(limits=limits)
        ax = env.fig.axes[0]
        try:
            ax.view_init(azim=azim, elev=elev)
        except NameError:
            pass # default view

        # robots
        env.add(self, jointlabels=True, jointaxes=False,
                    eeframe=True, shadow=False)

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
    """ Simple probe with ROT9 as the only degree of freedom. """

    ROT1, ROT2, ROT3 = 0., 0., 0.
    TRANS4, TRANS5, TRANS6 = 0., 0., 0.2

    constraints = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, None, 0.]

    def __init__(self, scope_length, theta_cam):
        """ Initialises probe as normal and sets the generalised
            coordinates, which are to be constrained, to the
            corresponding constant values. """

        super().__init__(scope_length, theta_cam)

        # redefine q and qdot (symbolic)
        constraints = self.__class__.constraints
        assert(len(constraints) == len(self.q_s))

        for i, c in enumerate(constraints):
            if c is not None:
                self.q_s[i] = c
                self.qd_s[i] = 0
                self.qdd_s[i] = 0

class RigidSimpleProbe(SimpleProbe):
    """ Simple probe with non-rotating, non-translating parts.
        Uses the BC configuration. """

    ROT1, ROT2, ROT3 = 0., 0., 0.
    TRANS4, TRANS5, TRANS6 = 0., 0., 0.2

    constraints = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, 0., 0.]

    def __init__(self, scope_length, theta_cam):
        super().__init__(scope_length, theta_cam)

        # set all joint values to 0
        self.q = [q if not isinstance(q, sp.Expr) else 0. for q in self.q_s]
        self.qd = [0] * self.n
        self.qdd = [0] * self.n

    @property
    def joint_dofs(self):
        return [*self.q, *self.qd, *self.qdd]

    @property
    def imu_dofs(self):
        return [*self.q[:6]]
