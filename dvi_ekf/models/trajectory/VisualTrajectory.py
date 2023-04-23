from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from dvi_ekf.models.measurement_point import VisualMeasurementPoint
from dvi_ekf.tools.Quaternion import Quaternion

from .TrajectoryBase import TrajectoryBase


class VisualTraj(TrajectoryBase):
    """Visual trajectory containing time and pose."""

    labels = ["t", "x", "y", "z", "qx", "qy", "qz", "qw"]

    def __init__(
        self,
        name: str,
        filepath: Optional[str] = None,
        notch_filepath: Optional[str] = None,
        cap: Optional[int] = None,
        scale: Optional[float] = None,
        with_notch: bool = False,
        start_at: Optional[int] = None,
    ):
        """
        :param name: name of the trajectory
        :param filepath: camera trajectory filepath
        :param notch_filepath: notch trajectory filepath
        :param cap: maximum number of frames
        :param scale: scaling factor
        :param with_notch: whether to incorporate notch trajectory or not
        :param start_at: which frame number to start the trajectory at
        """
        self.x: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.qx: Optional[np.ndarray] = None
        self.qy: Optional[np.ndarray] = None
        self.qz: Optional[np.ndarray] = None
        self.qw: Optional[np.ndarray] = None
        self.quats: Optional[np.ndarray] = None
        self.rx_deg: Optional[np.ndarray] = None
        self.ry_deg: Optional[np.ndarray] = None
        self.rz_deg: Optional[np.ndarray] = None

        super().__init__(name, filepath, cap)

        start_index: Optional[int] = (
            np.argwhere(self.t == start_at).item() if start_at else None
        )
        self._filter_values(self.labels, start_index, cap)
        self._scale_values(scale)

        self.labels_notch = ["notch", "notch_d", "notch_dd"]
        self.notch: Optional[np.ndarray] = None
        self.notch_d: Optional[np.ndarray] = None
        self.notch_dd: Optional[np.ndarray] = None
        self.notch_filepath = notch_filepath

        self.gen_angle_arrays()

        self.with_notch = with_notch
        if with_notch:
            self.read_notch_from_file()
            self._filter_values(self.labels_notch, start_index, cap)
            assert len(self.t) == len(self.notch)

    def _filter_values(
        self,
        labels: List[str],
        start_index: Optional[int] = None,
        cap: Optional[int] = None,
    ):
        """
        Filters the stored trajectory values by start index and max. values.
        :param labels: data labels/column headings
        :param start_index: start index of the data
        :param cap: maximum number of data points
        """
        if not start_index:
            if cap:
                index = 0
            else:
                return
        else:
            index: Optional[int] = np.argwhere(self.t == start_index).item()

        for label in labels:
            self.__dict__[label] = (
                self.__dict__[label][index:]
                if not cap
                else self.__dict__[label][index : index + cap]
            )

    def _scale_values(self, scale: Optional[float]) -> None:
        """
        Scales x, y, z measurement values.
        :param scale: scaling factor for the x, y, z values.
        """
        if not scale:
            return

        for label in ["x", "y", "z"]:
            self.__dict__[label] = [val * scale for val in self.__dict__[label]]

    def read_notch_from_file(self):
        with open(self.notch_filepath, "r") as f:
            for i, line in enumerate(f):
                data = line.split()

                # iterate over data containers
                for j, label in enumerate(self.labels_notch):
                    meas = float(data[j])
                    self.__dict__[label].append(meas)

    def at_index(self, index: int) -> VisualMeasurementPoint:
        """Returns single visual measurement at the given index."""
        t = self.t[index]

        x = self.x[index]
        y = self.y[index]
        z = self.z[index]

        qx = self.qx[index]
        qy = self.qy[index]
        qz = self.qz[index]
        qw = self.qw[index]
        q = Quaternion(x=qx, y=qy, z=qz, w=qw)

        return VisualMeasurementPoint(t, x, y, z, q)

    def gen_angle_arrays(self) -> None:
        """Updates angles."""
        if self.qx is None:
            return

        self._gen_quats_array()
        self._gen_euler_angles()

    def _gen_quats_array(self) -> None:
        """Generates self.quats from individual quaternion components."""
        self.quats = [
            Quaternion(x=self.qx[i], y=self.qy[i], z=self.qz[i], w=w, do_normalise=True)
            for i, w in enumerate(self.qw)
        ]

    def _gen_euler_angles(self):
        """
        Generates Euler angles from quaternions.
        extrinsic: xyz: rotations about fixed (global) CS
        intrinsic: XYZ: rotations about moving (local) CS
        """
        euler = np.array(
            [R.from_quat(q.xyzw).as_euler("xyz", degrees=True) for q in self.quats]
        )
        self.rx_deg = euler[:, 0]
        self.ry_deg = euler[:, 1]
        self.rz_deg = euler[:, 2]

        # euler = np.array([R.from_quat(q.xyzw).as_euler('XYZ', degrees=True) for q in self.quats])
        # self.rX_deg = euler[:,0]
        # self.rY_deg = euler[:,1]
        # self.rZ_deg = euler[:,2]

    @property
    def x_lims(self):
        return min(self.x), max(self.x)

    @property
    def y_lims(self):
        return min(self.y), max(self.y)

    @property
    def z_lims(self):
        return min(self.z), max(self.z)

    @property
    def lims(self):
        limits = [self.x_lims, self.y_lims, self.z_lims]
        return [val for min_max in limits for val in min_max]
