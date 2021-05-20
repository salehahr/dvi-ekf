import numpy as np
import quaternion # convenient for quat multiplication, normalisation
from scipy.spatial.transform import Rotation as R

class Quaternion(object):
    """ Quaternion convenience class. """

    def __init__(self, wxyz=None, xyzw=None, x=None, y=None, z=None, w=None, v=None, rot=None, do_normalise=False):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # parsing
        if wxyz is not None and type(wxyz) is quaternion.quaternion:
            self.w, self.x, self.y, self.z = quaternion.as_float_array(wxyz)
        elif xyzw is not None:
            self.x, self.y, self.z, self.w = xyzw
        elif rot is not None:
            self.x, self.y, self.z, self.w = R.from_matrix(rot).as_quat()
        elif v is not None:
            self.x, self.y, self.z = v

        # normalize
        if do_normalise:
            self.normalise()

    def __repr__(self):
        return f"Quaternion [x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, w={self.w:.3f}]"

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
    def np_quat(self):
        return np.quaternion(self.w, self.x, self.y, self.z)

    @property
    def euler_zyx(self):
        return R.from_quat(self.xyzw).as_euler('zyx')

    @property
    def conjugate(self):
        return Quaternion(wxyz=self.np_quat.conjugate())

    # operators
    def __mul__(self, other):
        if type(other) is quaternion.quaternion:
            mult = self.np_quat * other
        elif type(other) is Quaternion:
            mult = self.np_quat * other.np_quat
        else:
            mult = other * self.np_quat
        return Quaternion(wxyz=mult)

    def __rmul__(self, other):
        if isinstance(other, float):
            mult = other * self.np_quat
        return Quaternion(wxyz=mult)

    def __add__(self, other):
        summ = self.np_quat + other.np_quat
        return Quaternion(wxyz=summ)

    def normalise(self):
        quat = self.np_quat.normalized()

        # # constrain scalar part to be positive
        # w, _, _, _ = quaternion.as_float_array(quat)
        # if w < 0:
            # quat = -quat

        self.__init__(wxyz=quat)

