from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from scipy.spatial.transform import Rotation as R

from dvi_ekf.tools.Quaternion import Quaternion

from .TrajectoryBase import TrajectoryBase

if TYPE_CHECKING:
    from dvi_ekf.models.Imu import Imu


class ImuRefTraj(TrajectoryBase):
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

    def __init__(self, name: str, imu: Imu, filepath: Optional[str] = None):
        super().__init__(name, filepath)
        self._imu = imu
        # self.base = imu.imget_base()

    def append_value(
        self, t: float, cam_meas_vec: List[np.ndarray], current_notch: np.ndarray
    ):
        """Appends new measurement from current state."""

        p, R_WB, v = self._imu.ref_vals(cam_meas_vec, current_notch)

        euler_angs = R.from_matrix(R_WB).as_euler("xyz", degrees=True)
        quats = Quaternion(val=R_WB, do_normalise=True)
        data = [t, *p, *v, *euler_angs, *quats.wxyz]

        for i, label in enumerate(self.labels):
            if self.__dict__[label] is None:
                self.__dict__[label] = [data[i]]
            else:
                self.__dict__[label].append(data[i])
