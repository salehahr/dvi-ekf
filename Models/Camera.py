import numpy as np

from .Interpolator import Interpolator
from .context import Quaternion

from Filter import VisualTraj
from Visuals import CameraPlot

from copy import copy

class Camera(object):
    """ Class for the camera sensor which reads data from a text file.

        Provides trajectory data (positions p and rotations R or q)
        as well as the derivatives of the above
        (velocities: v and om,
        accelerations: acc and alp).

        Also provides the initial conditions.
    """

    def __init__(self, filepath, traj=None, max_vals=None, scale=None, notch=False):
        self.traj = traj    if (traj) else \
                    VisualTraj("camera", filepath, cap=max_vals,
                        scale=scale)
        self.max_vals = len(self.traj.t)

        self.t      = self.traj.t
        self.dt     = self.t[1] - self.t[0]
        self.min_t  = self.t[0]
        self.max_t  = self.t[-1]

        # measured data
        self.p      = np.array((self.traj.x, self.traj.y, self.traj.z))
        self.r      = np.array([q.euler_xyz_rad for q in self.traj.quats]).T
        self.R      = [q.rot for q in self.traj.quats]
        self.q      = np.array([q.xyzw for q in self.traj.quats]).T

        # derived data
        self.v = None
        self.acc = None
        self.om = None
        self.alp = None

        if not self.flag_interpolated:
            self._calc_derived_data()
        else:
            self._derived_data_from_traj()

        # initial conditions
        self.vec0   = self.vec_at(0)
        self.p0     = self.p[:,0].reshape(3,1)
        self.r0     = self.r[:,0].reshape(3,1)
        self.R0     = self.R[0]
        self.q0     = self.q[:,0]
        self.v0     = self.v[:,0].reshape(3,1)
        self.om0    = self.om[:,0].reshape(3,1)
        self.acc0   = self.acc[:,0].reshape(3,1)
        self.alp0   = self.alp[:,0].reshape(3,1)

        # notch
        self.is_rotated = not notch
        self.rotated = None

        if not self.flag_interpolated:
            self.notch = None
            self.notch_d = None
            self.notch_dd = None
        else:
            self.notch = self.traj.notch
            self.notch_d = self.traj.notch_d
            self.notch_dd = self.traj.notch_dd

        if not self.is_rotated:
            self._gen_notch_values()
            self.gen_rotated()

    @property
    def filepath(self):
        return self.traj.filepath

    @property
    def flag_interpolated(self):
        return self.traj.is_interpolated

    def interpolate(self, interframe_vals):
        interp_labels = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        interp_traj = Interpolator(interframe_vals, self).interpolated
        return CameraInterpolated(interp_traj)

    def _calc_derived_data(self):
        self.v = np.asarray( (np.gradient(self.p[0,:], self.dt),
                            np.gradient(self.p[1,:], self.dt),
                            np.gradient(self.p[2,:], self.dt)) )
        self.acc = np.asarray( (np.gradient(self.v[0,:], self.dt),
                            np.gradient(self.v[1,:], self.dt),
                            np.gradient(self.v[2,:], self.dt)) )

        ang_WC = np.asarray([q.euler_zyx_rad for q in self.traj.quats])
        rz, ry, rx = ang_WC[:,0], ang_WC[:,1], ang_WC[:,2]
        self.om = np.asarray( (np.gradient(rx, self.dt),
                            np.gradient(ry, self.dt),
                            np.gradient(rz, self.dt)) )
        self.alp = np.asarray( (np.gradient(self.om[0,:], self.dt),
                            np.gradient(self.om[1,:], self.dt),
                            np.gradient(self.om[2,:], self.dt)) )

    def _derived_data_from_traj(self):
        self.v = self.traj.v
        self.om = self.traj.om
        self.acc = self.traj.acc
        self.alp = self.traj.alp

    def gen_rotated(self):
        rotated_traj = VisualTraj("camera rot")
        rotated_traj.t = copy(self.t)
        rotated_traj.x = copy(self.traj.x)
        rotated_traj.y = copy(self.traj.y)
        rotated_traj.z = copy(self.traj.z)
        rotated_traj.quats = []

        for i, t in enumerate(self.t):
            real_quat = self.traj.quats[i]

            ang_notch = self.get_notch_at(i)[0]
            notch_quat = Quaternion(val=np.array([0, 0, ang_notch]), euler='xyz')

            rotated_quat = notch_quat * real_quat

            rotated_traj.quats.append(rotated_quat)
            rotated_traj.qx.append(rotated_quat.x)
            rotated_traj.qy.append(rotated_quat.y)
            rotated_traj.qz.append(rotated_quat.z)
            rotated_traj.qw.append(rotated_quat.w)

        rotated_traj.is_rotated = True
        rotated_traj._gen_euler_angles()
        self.rotated = Camera(filepath=None, traj=rotated_traj)

        self.rotated.notch = self.notch
        self.rotated.notch_d = self.notch_d
        self.rotated.notch_dd  = self.notch_dd

    def _gen_notch_values(self):
        def traj_gen(z0, zT, t, t_prev, T):
            t = t - t_prev
            T = T - t_prev

            z_n = z0 + (zT - z0) * (35*(t/T)**4 - 84*(t/T)**5 + 70*(t/T)**6 - 20*(t/T)**7)

            z_n_d = (zT - z0) * \
                    ( 35*4/(T**4)*t**3 - 84*5/(T**5)*t**4 \
                    + 70*6/(T**6)*t**5 - 20*7/(T**7)*t**6 )
            z_n_dd = (zT - z0) * \
                    ( 35*4*3/(T**4)*t**2 - 84*5*4/(T**5)*t**3 \
                    + 70*6*5/(T**6)*t**4 - 20*7*6/(T**7)*t**5 )

            return z_n, z_n_d, z_n_dd

        ang_vals = np.pi * np.array([0, 0, 0.9, 0.9, -0.9/2, -0.9/2])

        ang_prev = ang_vals[0]
        t_part = self._gen_t_partition()
        t_prev = t_part[0][0]

        traj = [0] * len(t_part)
        traj_d = [0] * len(t_part)
        traj_dd = [0] * len(t_part)

        for i, ang_k in enumerate(ang_vals[1:]):
            t_max = t_part[i][-1]
            traj[i], traj_d[i], traj_dd[i] = traj_gen(ang_prev, ang_k, t_part[i], t_prev, t_max)
            t_prev = t_max
            ang_prev = ang_k

        self.notch = np.concatenate(traj).ravel()
        self.notch_d = np.concatenate(traj_d).ravel()
        self.notch_dd = np.concatenate(traj_dd).ravel()

        self.traj.notch = np.concatenate(traj).ravel()
        self.traj.notch_d = np.concatenate(traj_d).ravel()
        self.traj.notch_dd = np.concatenate(traj_dd).ravel()

    def _gen_t_partition(self):
        partitions = np.array([0, 0.1, 0.45, 0.5, 0.9, 1])
        t_part = [0] * (len(partitions)-1)
        p_prev = 0
        for i, p in enumerate(partitions[1:]):
            p_k = int(np.ceil(p * self.max_vals))
            t_part[i] = np.array(self.t[p_prev : p_k])
            p_prev = p_k
        return t_part

    def get_notch_at(self, i):
        return [self.notch[i], self.notch_d[i], self.notch_dd[i]]

    def generate_queue(self, old_t, new_t):
        """ After old_t, up till new_t. """
        old_i = self._get_index_at(old_t)
        new_i = self._get_index_at(new_t)

        queue       = CameraQueue()
        queue.t     = self.t[old_i+1:new_i+1]
        queue.p     = self.p[:,old_i+1:new_i+1]
        queue.R     = self.R[old_i+1:new_i+1]
        queue.v     = self.v[:,old_i+1:new_i+1]
        queue.om    = self.om[:,old_i+1:new_i+1]
        queue.acc   = self.acc[:,old_i+1:new_i+1]
        queue.alp   = self.alp[:,old_i+1:new_i+1]

        return queue

    def at_index(self, i):
        return self.traj.at_index(i)

    def _get_index_at(self, T):
        """ Get index of camera data where the timestamp <= T. """
        return max([i for i, t in enumerate(self.t) if t <= T])

    def vec_at(self, i):
        p = self.p[:,i].reshape(3,1)
        R = self.R[i]
        v = self.v[:,i].reshape(3,1)
        om = self.om[:,i].reshape(3,1)
        acc = self.acc[:,i].reshape(3,1)
        alp = self.alp[:,i].reshape(3,1)

        return [p, R, v, om, acc, alp]

    def plot(self, config):
        CameraPlot(self).plot(config)

    def plot_notch(self, config):
        CameraPlot(self).plot_notch(config)

class CameraInterpolated(Camera):
    def __init__(self, traj):
        assert(traj.is_interpolated)
        super().__init__(filepath=None, traj=traj)

    @property
    def interframe_vals(self):
        return self.traj.interframe_vals

class CameraQueue(object):
    def __init__(self):
        pass

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def at_index(self, i):
        queue_item      = CameraQueue()
        queue_item.t    = self.t[i]
        queue_item.p    = self.p[:,i]
        queue_item.R    = self.R[i]
        queue_item.v    = self.v[:,i]
        queue_item.om   = self.om[:,i]
        queue_item.acc  = self.acc[:,i]
        queue_item.alp  = self.alp[:,i]
        return queue_item

    @property
    def vec(self):
        return [self.p, self.R, self.v, self.om, self.acc, self.alp]