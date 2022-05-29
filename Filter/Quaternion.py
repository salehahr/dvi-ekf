from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def raise_wrong_input(val):
    print(f"Wrong input type to Quaternion! Is: {type(val)}")
    raise TypeError


def euler_error_checking(e1, e2, verbosity=False):
    def ensure_close_deg(v1, v2):
        try:
            assert math.isclose(v1, v2, rel_tol=0.1, abs_tol=0.05)
        except AssertionError as e:
            if verbosity:
                print(f"Error in Euler angle: {v1:+0.3f} vs {v2:+0.3f}")

    x1, y1, z1 = e1
    z2, y2, x2 = e2
    ensure_close_deg(x1, x2)
    ensure_close_deg(y1, y2)
    ensure_close_deg(z1, z2)


class Quaternion(object):
    """Quaternion convenience class.
    Euler rotations are extrinsic: rotations about the fixed CS."""

    def __init__(
        self,
        val: Optional[np.ndarray, Quaternion] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        w: Optional[float] = None,
        v: Optional[np.ndarray] = None,
        do_normalise: bool = False,
        euler: str = "",
    ):
        """
        :param val: 3x3 rotation matrix, 4x1 quaternion vector (xyzw), or 3x1 xyz Euler angles
        :param x: x-component of quaternion
        :param y: y-component of quaternion
        :param z: z-component of quaternion
        :param w: scalar part of quaternion
        :param v: vector part of quaternion
        :param do_normalise: whether to perform normalisation
        :param euler: Euler convention, e.g. "xyz"
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        self._parse_inputs(val, x, y, z, w, v, euler)

        # normalise
        if do_normalise:
            self.normalise()

        # error checking
        do_check = False
        if do_check:
            euler_error_checking(self.euler_xyz_deg, self.euler_zyx_deg, verbosity=True)

    def __repr__(self):
        return f"Quaternion [x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, w={self.w:.3f}]"

    def _parse_inputs(
        self,
        val: Optional[np.ndarray, Quaternion] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        w: Optional[float] = None,
        v: Optional[np.ndarray] = None,
        euler: str = "",
    ) -> None:
        if val is not None:
            if isinstance(val, np.ndarray):
                if val.shape == (3, 3):
                    self.x, self.y, self.z, self.w = R.from_matrix(val).as_quat()
                elif val.shape == (4, 1) or val.shape == (4,):
                    self.x, self.y, self.z, self.w = val
                elif val.squeeze().shape == (3,) and euler == "xyz":
                    self.x, self.y, self.z, self.w = R.from_euler("xyz", val).as_quat()
                else:
                    print("Wrong dimensions of np.ndarray for Quaternion!")
                    raise ValueError
            elif isinstance(val, Quaternion):
                self.x = val.x
                self.y = val.y
                self.z = val.z
                self.w = val.w
            else:
                raise_wrong_input(val)
        elif v is not None and w is not None:
            # fmt: off
            self.x, self.y, self.z = v.reshape(3,)
            # fmt: on
        elif not None in [x, y, z, w]:
            pass
        else:
            raise_wrong_input(val)

    @property
    def v(self):
        # fmt: off
        return np.array([self.x, self.y, self.z]).reshape(3,)
        # fmt: on

    # rotation representations
    @property
    def sp_rot(self):
        """Scipy Rotation object"""
        return R.from_quat(self.xyzw)

    @property
    def rot(self):
        return self.sp_rot.as_matrix()

    @property
    def wxyz(self):
        return np.asarray([self.w, self.x, self.y, self.z])

    @property
    def xyzw(self):
        return np.asarray([self.x, self.y, self.z, self.w])

    @property
    def euler_xyz_rad(self):
        return self.sp_rot.as_euler("xyz", degrees=False)

    @property
    def euler_xyz_deg(self):
        return self.sp_rot.as_euler("xyz", degrees=True)

    @property
    def euler_zyx_rad(self):
        return self.sp_rot.as_euler("zyx", degrees=False)

    @property
    def euler_zyx_deg(self):
        return self.sp_rot.as_euler("zyx", degrees=True)

    @property
    def conjugate(self):
        return Quaternion(w=self.w, v=-self.v)

    @property
    def angle(self):
        """in radians"""
        # https://github.com/aipiano/ESEKF_IMU/blob/38320a8617cd3cf07231c9f6394b01755f7a5fff/esekf.py#L164
        ang1 = math.asin(np.linalg.norm(self.v))
        # ang2 = np.linalg.norm(self.axis)
        # ang3 = 2 * math.acos(self.w)
        # print(f'ang1 {ang1}')
        # print(f'ang2 {ang2}')
        # print(f'ang3 {ang3}')
        return ang1

    @property
    def axis(self):
        ang1 = math.asin(np.linalg.norm(self.v))
        ang3 = 2 * math.acos(self.w)

        ax1 = (
            np.zeros(
                3,
            )
            if math.isclose(ang1, 0)
            else self.v / np.linalg.norm(self.v)
        )
        # ax2 = self.sp_rot.as_rotvec()
        # ax3 = np.zeros(3,) if math.isclose(ang3, 0) else \
        # self.v / np.sin (ang3)
        # print(f'ax1 {ax1}')
        # print(f'ax2 {ax2}')
        # print(f'ax3 {ax3}')
        return ax1

    # operators
    def __mul__(self, other):
        if type(other) is Quaternion:
            w = self.w * other.w - self.v @ other.v
            v = self.w * other.v + other.w * self.v + np.cross(self.v, other.v)
        elif isinstance(other, float) or isinstance(other, int):
            w = other * self.w
            v = other * self.v
        elif isinstance(other, np.ndarray) and other.shape == (3, 3):
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
        d = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2 + self.w ** 2)
        self.x = self.x / d
        self.y = self.y / d
        self.z = self.z / d
        self.w = self.w / d

        # constrain scalar part to be positive
        if self.w < 0:
            xyzw = -self.xyzw
            self.__init__(val=xyzw)
