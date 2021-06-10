import roboticstoolbox as rtb
import spatialmath.base.symbolic as sym

import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
t = sp.symbols('t')

import matplotlib.pyplot as plt

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
    def __init__(self, scope_length, theta_cam, config='BC'):
        """ Initialises robot links. """
        self.config = config

        links = self._gen_links(config, scope_length, theta_cam)
        super().__init__(links, name='probe')

        self.q_sym = [dynamicsymbols(f"q{i+1}") for i in range(self.nlinks)]
        self.q_dot_sym = [sp.diff(q, t) for q in self.q_sym]
        self.q_ddot_sym = [sp.diff(q, t) for q in self.q_dot_sym]
        self.q_sym = [sp.Symbol(f'q{i+1}') for i in range(self.nlinks)]

    def _gen_links(self, config, scope_length, theta_cam):
        """ Generates robot links according to chosen configuration. """

        print(f"Initialising the probe in the configuration {config}...")

        if config == 'BC':
            links = [
                # imu orientation
                rtb.RevoluteDH(alpha=-sp.pi/2, offset=-sp.pi/2),
                rtb.RevoluteDH(alpha=sp.pi/2, offset=-sp.pi/2),
                rtb.RevoluteDH(alpha=sp.pi/2, offset=sp.pi/2),
                # imu translation
                rtb.PrismaticDH(theta=sp.pi/2, alpha=-sp.pi/2),
                rtb.PrismaticDH(theta=-sp.pi/2, alpha=-sp.pi/2),
                rtb.PrismaticDH(),
                # pivot rotations
                rtb.RevoluteDH(alpha=sp.pi/2, offset=sp.pi/2),
                rtb.RevoluteDH(alpha=sp.pi/2, offset=sp.pi/2),
                rtb.RevoluteDH(d=scope_length),
                # cam
                rtb.RevoluteDH(alpha=theta_cam),
                ]
        elif config == 'CB':
            links = [
                # angled tip
                rtb.RevoluteDH(alpha=-theta_cam),
                # rod to camera coupling
                rtb.RevoluteDH(alpha=-sp.pi/2, d=-scope_length),
                rtb.RevoluteDH(alpha=-sp.pi/2, offset=-sp.pi/2),
                rtb.RevoluteDH(alpha=0, offset=-sp.pi/2),
                # imu translation
                rtb.PrismaticDH(alpha=sp.pi/2, theta=0),
                rtb.PrismaticDH(alpha=sp.pi/2, theta=sp.pi/2),
                rtb.PrismaticDH(alpha=-sp.pi/2, theta=-sp.pi/2),
                # imu orientation
                rtb.RevoluteDH(alpha=-sp.pi/2, offset=-sp.pi/2),
                rtb.RevoluteDH(alpha=-sp.pi/2, offset=-sp.pi/2),
                rtb.RevoluteDH(),
                ]
        else:
            print("Invalid configuration!")

        return links

    @property
    def T(self):
        return self.fkine(self.q_sym)

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

    def plot(self, config, block=True):
        """ Modified to allow 'hold' of figure. """

        import tkinter
        print(f"Plotting robot with the configuration:\n\t {config}")

        # handles error when closing the window
        try:
            plt_obj = super().plot(config, block=block)
            return plt_obj
        except tkinter.TclError:
            return None

    ## ---  velocity calculations --- ##
    def _calc_velocity(self):
        J = self.jacob0(self.q_sym)
        return J @ self.q_dot_sym

    @property
    def v(self):
        v = self._calc_velocity()[:3]
        return dummify_array(v).reshape(3,1)

    @property
    def om(self):
        om = self._calc_velocity()[-3:]
        return dummify_array(om).reshape(3,1)

    ## --- acceleration calculations --- ##
    def _calc_acceleration(self):
        J = self.jacob0(self.q_sym)
        H = self.hessian0(q=self.q_sym)
        return H @ self.q_dot_sym @ self.q_dot_sym + J @ self.q_ddot_sym

    @property
    def acc(self):
        a = self._calc_acceleration()[:3]
        return dummify_array(a).reshape(3,1)

    @property
    def alp(self):
        alp = self._calc_acceleration()[-3:]
        return dummify_array(alp).reshape(3,1)

    ## --- reversed kinematic relations --- ##
    def get_reversed_kin(self):
        """ Get reversed kinematics relations (i.e. B relative to C) """
        om_p_cross = np.cross(self.om, self.p,axis=0)

        p_rev = -self.p
        v_rev = -self.v + om_p_cross
        acc_rev = -self.acc \
                    + np.cross(self.acc, self.p, axis=0) \
                    + 2 * np.cross(self.om, self.v, axis=0) \
                    - np.cross(self.om, om_p_cross, axis=0)

        om_rev = -self.om
        alp_rev = -self.alp

        return p_rev, v_rev, acc_rev, om_rev, alp_rev

class SimpleProbe(Probe):
    """ Simple probe with ROT9 as the only degree of freedom. """

    ROT1, ROT2, ROT3 = 0., 0., 0.
    TRANS4, TRANS5, TRANS6 = 0., 0., 0.2
    ROT7, ROT8 = 0., 0.

    constraints_BC = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, ROT7, ROT8,
                    None, 0.]

    constraints_CB = [0.,
                    None, ROT8, ROT7,
                    -TRANS6, TRANS5, TRANS4,
                    ROT3, ROT2, ROT1
                    ]

    def __init__(self, scope_length, theta_cam, config='BC'):
        """ Initialises probe as normal and sets the generalised
            coordinates, which are to be constrained, to the
            corresponding constant values. """

        super().__init__(scope_length, theta_cam, config)

        # redefine q and qdot (symbolic)
        constraints = eval(f"self.__class__.constraints_{config}")
        assert(len(constraints) == len(self.q_sym))

        for i, c in enumerate(constraints):
            if c is not None:
                self.q_sym[i] = c
        self.q_dot_sym = [sp.diff(q, t) for q in self.q_sym]

class RigidSimpleProbe(SimpleProbe):
    """ Simple probe with non-rotating, non-translating parts.
        Uses the BC configuration. """

    ROT1, ROT2, ROT3 = 0., 0., 0.
    TRANS4, TRANS5, TRANS6 = 0., 0., 0.2
    ROT7, ROT8, ROT9 = 0., 0., 0.

    constraints_BC = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, ROT7, ROT8,
                    ROT9, 0.]

    def __init__(self, scope_length, theta_cam):
        super().__init__(scope_length, theta_cam, 'BC')

        # set all joint values to 0
        self.q = [q if not isinstance(q, sp.Expr) else 0. for q in self.q_sym]
        self.qd = [sp.diff(q, t) for q in self.q_sym]
        self.qdd = [sp.diff(q, t) for q in self.q_dot_sym]

    @property
    def joint_dofs(self):
        return [*self.q, *self.qd, *self.qdd]
