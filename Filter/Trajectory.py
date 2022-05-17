import os
import sys
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from Filter.Quaternion import Quaternion
from Filter.States import process_data
from tools import files


class Trajectory(object):
    """Base trajectory class which requires a
    trajectory name, trajectory labels, and
    a filepath.
    """

    labels: List[str] = ["t"]

    def __init__(
        self,
        name: str,
        filepath: Optional[str] = None,
        cap: Optional[int] = None,
    ):
        """
        :param name: name of the trajectory
        :param filepath: camera trajectory filepath
        :param cap: maximum number of data
        """
        self.name = name
        self.filepath = filepath
        self.cap = cap

        # volatile values (can change after initialisation)
        self._is_interpolated = False
        self._interframe_vals = 1

        self.t: Optional[np.ndarray] = None

        self.reset()
        self._parse_if_file_exists(filepath, cap)

    def reset(self):
        """Reinitialises data labels."""
        for label in self.labels:
            self.__dict__[label] = None

    @property
    def nvals(self) -> int:
        """Number of values stored."""
        return len(self.t)

    @property
    def is_interpolated(self) -> bool:
        return self._is_interpolated

    @is_interpolated.setter
    def is_interpolated(self, val: bool) -> None:
        self._is_interpolated = val

    @property
    def interframe_vals(self) -> int:
        """Number of interpolated values between frames (F0 < interpolated_data <= F1)"""
        return self._interframe_vals

    @interframe_vals.setter
    def interframe_vals(self, val: int) -> None:
        self._interframe_vals = val

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def _parse_if_file_exists(self, filepath: str, cap: int) -> None:
        """
        Parse trajectory from file if file is given and file exists.

        :param filepath: trajectory filepath
        :param cap: maximum number of measurement values
        :return:
        """
        if self.filepath is None:
            return

        if os.path.exists(self.filepath):
            self._parse(cap)
        else:
            ans = input(f"File '{filepath}' not found. Create? (Y/N)")

            if ans.lower() == "y":
                file = open(filepath, "w+")
                file.close()
            else:
                sys.exit()

    def _parse(self, max_vals: int) -> None:
        """
        Extract data from file.
        :param max_vals: maxinum number of data to store
        """
        df = files.parse(self.filepath, max_vals)
        for label in self.labels:
            self.__dict__[label] = df[label].to_numpy()

    def _get_index_at(self, T: float) -> int:
        """
        Get index for which timestamp matches the argument T.
        :param T: time value to match
        :return: index of the timestamp
        """
        return max([i for i, t in enumerate(self.t) if t <= T])

    def write_to_file(self, filename: str) -> None:
        """
        Writes trajectory to disk.
        :param filename: trajectory filepath.
        """
        with open(filename, "w+") as f:
            data_str = ""

            for n in range(self.nvals):
                for label in self.labels:
                    data = self.__dict__[label][n]
                    if label == "t":
                        data_str += f"{data:.6f}"
                    else:
                        data_str += f" {data:.9f}"
                data_str += " \n"
            f.write(data_str)


class ImuRefTraj(Trajectory):
    """Desired traj of the IMU."""

    labels = [
        "t",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "rx",
        "ry",
        "rz",
        "qw",
        "qx",
        "qy",
        "qz",
    ]

    def __init__(self, name, imu, filepath=None):
        super().__init__(name, filepath)
        self.imu = imu

    def append_value(self, t, current_cam, current_notch):
        """Appends new measurement from current state."""

        p, R_WB, v = self.imu.ref_vals(current_cam, current_notch)

        euler_angs = R.from_matrix(R_WB).as_euler("xyz", degrees=True)
        quats = Quaternion(val=R_WB, do_normalise=True)
        data = [t, *p, *v, *euler_angs, *quats.wxyz]

        for i, label in enumerate(self.labels):
            if self.__dict__[label] is None:
                self.__dict__[label] = [data[i]]
            else:
                self.__dict__[label].append(data[i])


class FilterTraj(Trajectory):
    labels_imu = [
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "rx",
        "ry",
        "rz",
        "qw",
        "qx",
        "qy",
        "qz",
    ]
    labels_imu_dofs = ["dof1", "dof2", "dof3", "dof4", "dof5", "dof6"]
    labels_camera = [
        "xc",
        "yc",
        "zc",
        "rx_degc",
        "ry_degc",
        "rz_degc",
        "qwc",
        "qxc",
        "qyc",
        "qzc",
    ]
    labels = ["t", *labels_imu, *labels_imu_dofs, *labels_camera]

    def __init__(self, name, filepath=None):
        self.labels = self.labels
        self.labels_imu = self.labels_imu
        self.labels_imu_dofs = self.labels_imu_dofs
        self.labels_camera = self.labels_camera

        self.rx: Optional[np.ndarray] = None
        self.ry: Optional[np.ndarray] = None
        self.rz: Optional[np.ndarray] = None
        self.rx_degc: Optional[np.ndarray] = None
        self.ry_degc: Optional[np.ndarray] = None
        self.rz_degc: Optional[np.ndarray] = None

        super().__init__(name, filepath)

    def append_propagated_states(self, t, state):
        data = process_data(t, state)
        for i, label in enumerate(self.labels):
            if self.__dict__[label] is None:
                self.__dict__[label] = [data[i]]
            else:
                self.__dict__[label].append(data[i])

    def append_updated_states(self, t, state):
        data = process_data(t, state)
        for i, label in enumerate(self.labels):
            self.__dict__[label][-1] = data[i]

    @property
    def rz_unwrapped(self):
        return np.unwrap(self.rz)

    @property
    def ry_unwrapped(self):
        return np.unwrap(self.ry)

    @property
    def rx_unwrapped(self):
        return np.unwrap(self.rx)

    @property
    def rz_degc_unwrapped(self):
        return np.unwrap(self.rz_degc)

    @property
    def ry_degc_unwrapped(self):
        return np.unwrap(self.ry_degc)

    @property
    def rx_degc_unwrapped(self):
        return np.unwrap(self.rx_degc)
