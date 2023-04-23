from __future__ import annotations

from typing import List, Optional

import numpy as np

from dvi_ekf.filter.state import State

from .TrajectoryBase import TrajectoryBase


def _get_euler_measurement_array(t: float, state: State) -> List[float]:
    """
    Returns a float array containing measurement values where the angles are Euler xyz.
    :param t: current time measured
    :param state: current state measured
    :return: list of measurement values
    """
    dof_rots_deg = np.rad2deg(state.dofs[:3])
    dof_trans = state.dofs[3:]
    return [
        t,
        *state.p,
        *state.v,
        *state.q.euler_xyz_deg,
        *state.q.wxyz,
        *dof_rots_deg,
        *dof_trans,
        *state.p_cam,
        *state.q_cam.euler_xyz_deg,
        *state.q_cam.wxyz,
    ]


class FilterTraj(TrajectoryBase):
    # fmt: off
    labels_imu = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "qw", "qx", "qy", "qz"]
    labels_imu_dofs = ["dof1", "dof2", "dof3", "dof4", "dof5", "dof6"]
    labels_camera = ["xc", "yc", "zc", "rx_degc", "ry_degc", "rz_degc", "qwc", "qxc", "qyc", "qzc"]
    labels = ["t", *labels_imu, *labels_imu_dofs, *labels_camera]
    # fmt: on

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
        data = _get_euler_measurement_array(t, state)
        for i, label in enumerate(self.labels):
            if self.__dict__[label] is None:
                self.__dict__[label] = [data[i]]
            else:
                self.__dict__[label].append(data[i])

    def append_updated_states(self, t, state):
        data = _get_euler_measurement_array(t, state)
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
