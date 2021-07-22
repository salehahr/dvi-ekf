import roboticstoolbox as rtb
from spatialmath import SE3
import spatialmath.base.symbolic as sym

from casadi import *
import numpy as np
import sympy as sp
from sympy.tensor.array import tensorproduct, tensorcontraction

from . import context
from symbols import q_s, qd_s, qdd_s, dofs_s
from symbols import q_cas, q_tr_cas, dofs_cas_list

from aux_symbolic import dummify_array

from roboticstoolbox.backends.PyPlot import PyPlot
import tkinter
import matplotlib.pyplot as plt

# just so that the plot is orientated correctly...
plot_rotation = SE3.Ry(-180, 'deg')

class Probe(rtb.DHRobot):
    def __init__(self, scope_length, theta_cam):
        """ Initialises robot links. """

        links = self._gen_links(scope_length, theta_cam)
        super().__init__(links, name='probe')

        self.scope_length = scope_length
        self.theta_cam = theta_cam

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
    def q_cas(self):
        return self._to_casadi(self.q_s)
    @property
    def qd_cas(self):
        return casadi.SX(self.qd_s)
    @property
    def qdd_cas(self):
        return casadi.SX(self.qdd_s)

    @property
    def qd_s_arr(self):
        return sp.MutableDenseNDimArray(self.qd_s, (self.n, 1))

    @property
    def T(self):
        return self.fkine(self.q_s)

    @property
    def R(self):
        R = self.T.R
        return self._to_casadi(R)

    @property
    def p(self):
        p = self.T.t
        return self._to_casadi(p)

    ## ---  velocity calculations --- ##
    def _calc_velocity(self):
        J = self.jacob0(self.q_s)
        return J @ self.qd_s

    @property
    def v(self):
        v = self._calc_velocity()[:3]
        return self._to_casadi(v)

    @property
    def om(self):
        om = self._calc_velocity()[-3:]
        return self._to_casadi(om)

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
        tp2 = tp2.reshape(6,)

        # H @ qd @ qd + J @ qdd
        res = tp2 + J0 @ self.qdd_s
        # res = sp.Array(tp2 + (J0 @ self.qdd_s).reshape(6,1))
        return res

    @property
    def acc(self):
        acc = self._calc_acceleration()[:3]
        return self._to_casadi(acc)

    @property
    def alp(self):
        alp = self._calc_acceleration()[-3:]
        return self._to_casadi(alp)

    @property
    def joint_dofs(self):
        return [self.q_cas, self.qd_cas, self.qdd_cas]

    @property
    def imu_dofs(self):
        return [*self.q[:6]]

    def _to_casadi(self, var):
        if isinstance(var, np.ndarray):
            is_1dim = var.ndim == 1
            cs = casadi.SX(var.shape[0], 1) if is_1dim \
                    else casadi.SX(*var.shape)
        elif isinstance(var, list):
            is_1dim = True
            cs = casadi.SX(len(var), 1)

        if is_1dim:
            for i, v in enumerate(var):
                if isinstance(v, sp.Expr):
                    f = sp.lambdify(dofs_s, dummify_array(v))
                    cs[i,0] = f(*dofs_cas_list)
                else:
                    cs[i,0] = v
        else:
            for i, r in enumerate(var):
                for j, c in enumerate(r):
                    f = sp.lambdify(dofs_s, dummify_array(c))
                    cs[i,j] = f(*dofs_cas_list)

        return cs

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

    def plot(self, config, block=True, limits=None, movie=None, is_static=True, dt=0.05):
        """ Modified to allow 'hold' of figure. """

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

        # base
        self.base = plot_rotation

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
                    plt.figure(env.fig)
                    plt.ioff()
            except tkinter.TclError:
                # handles error when closing the window
                return None

        # reset base
        self.base = None

    def _plot_frames(self, cam=None, ax=None):
        # base
        self.base.plot(frame='B', arrow=False, axes=ax, length=0.1, color='black')

        # pivot
        P = self.fkine_all(self.q_s)[6]
        P_t = np.eye(4)
        P_t[:3, :3] = P.R
        P_t[:3, -1] = P.t
        P = SE3(P_t)
        P.plot(frame='P', arrow=False, axes=ax, length=0.1, color='black')

        # virtual slam
        if cam:
            B_p_cam = casadi.DM(self.p + cam.p).full()
            C = np.eye(4)
            C[:3, :3] = casadi.DM(self.R).full()
            C[:3, -1] = B_p_cam[:,0]
            C = SE3(C)
            C.plot(frame='C', arrow=False, axes=ax, length=0.1, color='black')

    def plot_with_kf_traj(self, cam=None, imu_ref=None, kf_traj=None,
        filename='', limits=None, dt=0.05, azim=-37, elev=29):
        """ Note: this plot shows everything in the B coordinate system.
            The base only transforms the whole thing so that the camera
            starts at zero.

            imu_ref.base: transforms coords in B to W
                e.g. coords_in_W = imu_ref.base @ coords_in_B
            imu_ref.base.inv(): transforms coords in W to B
                e.g. coords_in_B = imu_ref.base.inv() @ coords_in_W
        """

        print(f"Plotting robot with the configuration:\n\t {self.q_s}")

        env = PyPlot()

        # visuals
        limit_x = [-0.2, 0.2]
        limit_y = [-0.2, 0.7]
        limit_z = [-0.3, 0.1]
        limits = [*limit_x, *limit_y, *limit_z] if (limits is None) else limits

        env.launch(limits=limits)
        ax = env.fig.axes[0]
        try:
            ax.view_init(azim=azim, elev=elev)
        except NameError:
            pass # default view

        # base
        orig_base = self.base
        plot_rotation = SE3.Rx(90, 'deg') @ SE3.Rx(self.theta_cam, 'rad')
        if imu_ref.base:
            # view everything in W, with a transf. on top
            self.base = plot_rotation @ imu_ref.base @ orig_base

            # view everything in W
            # self.base = imu_ref.base @ orig_base

            # view everything in B
            # self.base = None

        # robots
        env.add(self, jointlabels=True, jointaxes=False,
                    eeframe=True, shadow=False)
        self.q = self.q_s
        env.step(dt)

        def transform_trajectories(traj=None, x=None, y=None, z=None, n=None):
            """ Transforms coords from W to new base due to plot_rotation. """
            if traj:
                coords_w_rot = [plot_rotation @ SE3(traj.x[i],
                                    traj.y[i], traj.z[i])
                                for i in range(n)]
            elif x and y and z:
                coords_w_rot = [plot_rotation @ SE3(x[i], y[i], z[i])
                                for i in range(n)]
            else:
                print('Invalid input to function transform_trajectories')
                raise Exception

            coords_w_rot = [coords_w_rot[i].t for i in range(n)]
            return np.concatenate(coords_w_rot).reshape(n,3).T

        # trajectories
        if cam:
            # coords in B
            # B_p_cam = casadi.DM(self.p + cam.p).full()

            # coords in W
            cam_w_rot = transform_trajectories(traj=cam.traj, n=cam.max_vals)
            ax.plot(cam_w_rot[0,:], cam_w_rot[1,:], cam_w_rot[2,:], label='cam ref')

        if imu_ref:
            # coords in B
            # imu_b = [imu_ref.base.inv() @ SE3(imu_ref.x[n], imu_ref.y[n], imu_ref.z[n]) for n in range(imu_ref.n)]

            # coords in W, taking into account plot_rotation
            imu_ref_w_rot = transform_trajectories(traj=imu_ref,
                                    n=imu_ref.nvals)
            ax.plot(imu_ref_w_rot[0,:], imu_ref_w_rot[1,:], imu_ref_w_rot[2,:], label='imu ref')

        if kf_traj:
            imu_w_rot = transform_trajectories(traj=kf_traj, n=kf_traj.nvals)
            ax.plot(imu_w_rot[0,:], imu_w_rot[1,:], imu_w_rot[2,:], label='imu est')

            cam_w_rot = transform_trajectories(x=kf_traj.xc,
                                y=kf_traj.yc,
                                z=kf_traj.zc,
                                n=cam.max_vals)
            ax.plot(cam_w_rot[0,:], cam_w_rot[1,:], cam_w_rot[2,:], label='cam est')

        # frames
        self._plot_frames(cam=cam, ax=env.fig.axes[0])

        plt.figure(env.fig)
        plt.ioff()
        plt.legend()

        # reset base
        self.base = orig_base

        # save img
        if filename:
            plt.savefig(filename, dpi=200)

        plt.show()

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
                self.q[i] = c
                self.q_s[i] = c

                self.qd[i] = 0
                self.qd_s[i] = 0

                self.qdd[i] = 0
                self.qdd_s[i] = 0

class RigidSimpleProbe(SimpleProbe):
    """ Simple probe with non-rotating, non-translating parts.
        Uses the BC configuration. """

    ROT1, ROT2, ROT3 = 0., 0., 0.
    TRANS4, TRANS5, TRANS6 = 0., 0., 0.2

    constraints = [ROT1, ROT2, ROT3, TRANS4, TRANS5, TRANS6, 0., 0.]

    def __init__(self, scope_length, theta_cam):
        super().__init__(scope_length, theta_cam)

        # set all joint values to 0 except for the given constant values
        self.q = [q if not isinstance(q, sp.Expr) else 0. for q in self.q_s]
        self.qd = [0] * self.n
        self.qdd = [0] * self.n

class SymProbe(object):
    """ Container class for probe that only stores the
        symbolic forward kinematics relations. """

    def __init__(self, probe):
        self.n = probe.n
        self.q0 = probe.q.copy()
        self.q = q_s.copy()

        for i in range(6,self.n):
            self.q[i] = 0

        p, R, v, om, acc, alp = probe.get_sym(self.q)
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

    def _get_tr(self, expr):
        """ Returns kinematic expressions in terms of
            estimated DOFs and error DOFs."""
        return casadi.substitute(expr, q_cas, q_tr_cas)
