import numpy as np
from Filter import VisualTraj

class Camera(object):
    """ Class for the camera sensor which reads data from a text file.

        Provides trajectory data (positions p and rotations R or q)
        as well as the derivatives of the above
        (velocities: v and om,
        accelerations: acc and alp).

        Also provides the initial conditions of p, q, v, om.
    """

    def __init__(self, filepath, traj=None, max_vals=None):
        self.traj_filepath = filepath
        self.traj = traj if (traj) else \
            VisualTraj("camera", self.traj_filepath, cap=max_vals)
        self.max_vals = len(self.traj.t)

        self.t = self.traj.t
        self.dt = self.t[1] - self.t[0]

        self._p = None
        self._r = None
        self._R = None
        self._q = None
        self._v = None
        self._om = None
        self._acc = None
        self._alp = None

        self.notch = None
        self.notch_d = None
        self.notch_dd = None
        if self.max_vals >= 10:
            self._gen_notch_values()

        self.p0 = self.p[:,0].reshape(3,1)
        self.r0 = self.r[:,0].reshape(3,1)
        self.R0 = self.R[0]
        self.q0 = self.q[:,0]
        self.v0 = self.v[:,0].reshape(3,1)
        self.om0 = self.om[:,0].reshape(3,1)
        self.acc0 = self.acc[:,0].reshape(3,1)
        self.alp0 = self.alp[:,0].reshape(3,1)

        self._flag_interpolated = False
        self._num_imu_between_frames = 0

    @property
    def p(self):
        self._p = np.array((self.traj.x, self.traj.y, self.traj.z))
        return self._p

    @property
    def v(self):
        self._v = np.asarray( (np.gradient(self.p[0,:], self.dt),
                            np.gradient(self.p[1,:], self.dt),
                            np.gradient(self.p[2,:], self.dt)) )
        return self._v

    @property
    def acc(self):
        self._acc = np.asarray( (np.gradient(self.v[0,:], self.dt),
                            np.gradient(self.v[1,:], self.dt),
                            np.gradient(self.v[2,:], self.dt)) )
        return self._acc

    @property
    def r(self):
        self._r = np.array([q.euler_xyz for q in self.traj.quats]).T
        return self._r

    @property
    def R(self):
        self._R = [q.rot for q in self.traj.quats]
        return self._R

    @property
    def q(self):
        self._q = np.array([q.xyzw for q in self.traj.quats]).T
        return self._q

    @property
    def om(self):
        ang_WC = np.asarray([q.euler_zyx for q in self.traj.quats])
        rz, ry, rx = ang_WC[:,0], ang_WC[:,1], ang_WC[:,2]

        self._om = np.asarray( (np.gradient(rx, self.dt),
                            np.gradient(ry, self.dt),
                            np.gradient(rz, self.dt)) )
        return self._om

    @property
    def alp(self):
        self._alp = np.asarray( (np.gradient(self.om[0,:], self.dt),
                            np.gradient(self.om[1,:], self.dt),
                            np.gradient(self.om[2,:], self.dt)) )
        return self._alp

    @property
    def flag_interpolated(self):
        self._flag_interpolated = self.traj.flag_interpolated
        return self._flag_interpolated

    @property
    def num_imu_between_frames(self):
        self._num_imu_between_frames = self.traj.num_imu_between_frames
        return self._num_imu_between_frames

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

    def interpolate(self, num_imu_between_frames):
        self.traj.interpolate(num_imu_between_frames)
        traj_interp = self.traj.interpolated
        return Camera(filepath='', traj=self.traj.interpolated)

    def generate_queue(self, old_t, new_t):
        """ After old_t, up till new_t. """
        old_i = self._get_index_at(old_t)
        new_i = self._get_index_at(new_t)

        queue = Queue()
        queue.t = self.t[old_i+1:new_i+1]
        queue.p = self.p[:,old_i+1:new_i+1]
        queue.R = self.R[old_i+1:new_i+1]
        queue.v = self.v[:,old_i+1:new_i+1]
        queue.om = self.om[:,old_i+1:new_i+1]
        queue.acc = self.acc[:,old_i+1:new_i+1]
        queue.alp = self.alp[:,old_i+1:new_i+1]

        return queue

    def _get_index_at(self, T):
        """ Get index of camera data whose timestamp matches the argument T. """
        return max([i for i, t in enumerate(self.t) if t <= T])

class Queue(object):
    def __init__(self):
        pass

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def at_index(self, i):
        queue_item = Queue()
        queue_item.t = self.t[i]
        queue_item.p = self.p[:,i]
        queue_item.R = self.R[i]
        queue_item.v = self.v[:,i]
        queue_item.om = self.om[:,i]
        queue_item.acc = self.acc[:,i]
        queue_item.alp = self.alp[:,i]
        return queue_item