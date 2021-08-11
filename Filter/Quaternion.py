import numpy as np
import math
# import quaternion # convenient for quat multiplication, normalisation
from scipy.spatial.transform import Rotation as R

def skew(x):
    return np.array([[0,    -x[2], x[1]],
                     [x[2],    0, -x[0]],
                     [-x[1], x[0],    0]])

class Quaternion(object):
    """ Quaternion convenience class. """

    def __init__(self, val=None, x=None, y=None, z=None, w=None, v=None, do_normalise=False, euler=''):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # parsing
        if val is not None:
            if isinstance(val, np.ndarray):
                if val.shape == (3,3):
                    self.x, self.y, self.z, self.w = R.from_matrix(val).as_quat()
                elif val.shape == (4,1) or val.shape == (4,):
                    self.x, self.y, self.z, self.w = val
                elif val.squeeze().shape == (3,) and euler == 'xyz':
                    self.x, self.y, self.z, self.w = R.from_euler('xyz', val).as_quat()
                else:
                    print("Wrong dimensions of np.ndarray for Quaternion!")
                    raise ValueError
            elif isinstance(val, Quaternion):
                self.x = val.x
                self.y = val.y
                self.z = val.z
                self.w = val.w
            else:
                print("Wrong input type to Quaternion!")
                raise TypeError

        elif v is not None and w is not None:
            self.x, self.y, self.z = v.reshape(3,)

        elif x is not None and y is not None and z is not None and w is not None:
            pass
        else:
            print("Wrong input type to Quaternion!")
            raise TypeError

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
    def euler_xyz_deg(self):
        return R.from_quat(self.xyzw).as_euler('xyz', degrees=True)

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
        elif isinstance(other, np.ndarray) and other.shape == (3,3):
            q_other = Quaternion(val=other, do_normalise=True)
            w = self.w * q_other.w - self.v @ q_other.v
            v = self.w * q_other.v + q_other.w * self.v + np.cross(self.v, q_other.v)
        else:
            raise TypeError
        return Quaternion(w=w, v=v, do_normalise=True)

    def __add__(self, other):
        w = self.w + other.w
        v = self.v + other.v
        return Quaternion(w=w, v=v, do_normalise=True)

    def __sub__(self, other):
        w = self.w - other.w
        v = self.v - other.v
        return Quaternion(w=w, v=v, do_normalise=True)

    def normalise(self):
        d = math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
        self.x = self.x/d
        self.y = self.y/d
        self.z = self.z/d
        self.w = self.w/d

        # constrain scalar part to be positive
        if self.w < 0:
            xyzw = -self.xyzw
            self.__init__(val=xyzw)

