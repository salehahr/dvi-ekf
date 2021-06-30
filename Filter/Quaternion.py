import numpy as np
import math
# import quaternion # convenient for quat multiplication, normalisation
from scipy.spatial.transform import Rotation as R

class Quaternion(object):
    """ Quaternion convenience class. """

    def __init__(self, xyzw=None, x=None, y=None, z=None, w=None, v=None, rot=None, do_normalise=False):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # parsing
        if xyzw is not None:
            self.x, self.y, self.z, self.w = xyzw
        elif rot is not None:
            self.x, self.y, self.z, self.w = R.from_matrix(rot).as_quat()
        elif v is not None:
            self.x, self.y, self.z = v.reshape(3,)

        # normalize
        if do_normalise:
            self.normalise()

    def __repr__(self):
        return f"Quaternion [x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, w={self.w:.3f}]"

    @property
    def v(self):
        return np.array([self.x, self.y, self.z]).reshape(3,)

    # rotation representations
    @property
    def R(self):
        return R.from_quat(self.xyzw)

    @property
    def rot(self):
        return R.from_quat(self.xyzw).as_matrix()

    @property
    def wxyz(self):
        return np.asarray([self.w, self.x, self.y, self.z])

    @property
    def xyzw(self):
        return np.asarray([self.x, self.y, self.z, self.w])

    @property
    def euler_xyz(self):
        return R.from_quat(self.xyzw).as_euler('xyz')

    @property
    def euler_zyx(self):
        return R.from_quat(self.xyzw).as_euler('zyx')

    @property
    def conjugate(self):
        return Quaternion(w=self.w, v=-self.v)


    # operators
    def __mul__(self, other):
        if type(other) is Quaternion:
            w = self.w * other.w - self.v @ other.v
            v = self.w * other.v + other.w * self.v + np.cross(self.v, other.v)
        elif isinstance(other, float) or isinstance(other, int):
            w = other * self.w
            v = other * self.v
        else:
            raise TypeError
        return Quaternion(w=w, v=v)

    def __add__(self, other):
        w = self.w + other.w
        v = self.v + other.v
        return Quaternion(w=w, v=v)

    def __sub__(self, other):
        w = self.w - other.w
        v = self.v - other.v
        return Quaternion(w=w, v=v)

    def normalise(self):
        d = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        self.x = self.x/d
        self.y = self.y/d
        self.z = self.z/d
        self.w = self.w/d

        # # constrain scalar part to be positive
        # w, _, _, _ = quaternion.as_float_array(quat)
        # if w < 0:
            # quat = -quat

        # self.__init__(wxyz=quat)

