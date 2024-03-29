from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from spatialmath import SE3, UnitQuaternion

from dvi_ekf.models.measurement_point import CameraMeasurementPoint
from dvi_ekf.models.trajectory.Interpolator import Interpolator
from dvi_ekf.models.trajectory.VisualTrajectory import VisualTraj
from dvi_ekf.tools.Quaternion import Quaternion
from dvi_ekf.visuals import CameraPlot

if TYPE_CHECKING:
    from dvi_ekf.tools.config import Config


def create_camera(config: Config) -> Camera:
    """Creates new camera object from the config object."""
    cam = Camera(
        filepath=config.traj_fp,
        notch_filepath=config.notch_fp,
        max_vals=config.max_vals,
        scale=config.camera.scale,
        with_notch=config.with_notch,
        start_at=config.camera.start_frame,
    )

    if config.with_notch:
        assert cam.rotated is not None
    else:
        assert cam.rotated is None

    cam.update_sim_params(config)

    return cam


class Camera(object):
    """Class for the camera sensor which reads data from a text file.

    Provides trajectory data (positions p and rotations R or q)
    as well as the derivatives of the above
    (velocities: v and om,
    accelerations: acc and alp).

    Also provides the initial conditions.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        notch_filepath: Optional[str] = None,
        traj: Optional[VisualTraj] = None,
        max_vals: Optional[int] = None,
        scale: Optional[float] = None,
        with_notch: bool = False,
        start_at=None,
    ):
        """
        :param filepath: camera trajectory filepath
        :param notch_filepath: notch trajectory filepath
        :param traj: VisualTraj object
        :param max_vals: maximum number of frames
        :param scale: scaling factor
        :param with_notch: whether to incorporate notch trajectory or not
        :param start_at: which frame number to start the trajectory at
        """
        self.traj = (
            traj
            if traj
            else VisualTraj(
                "camera",
                filepath,
                notch_filepath=notch_filepath,
                cap=max_vals,
                scale=scale,
                with_notch=with_notch,
                start_at=start_at,
            )
        )
        self.max_vals = len(self.traj.t)

        self.t: np.ndarray = self.traj.t
        self.dt: float = self.t[1] - self.t[0]
        self.min_t: float = self.t[0]
        self.max_t: float = self.t[-1]

        # measured data
        self.p = np.array((self.traj.x, self.traj.y, self.traj.z))
        self.r = np.array([q.euler_xyz_rad for q in self.traj.quats]).T
        self.R = [q.rot for q in self.traj.quats]
        self.q = np.array([q.xyzw for q in self.traj.quats]).T

        # frames
        rots_vec = SE3([UnitQuaternion(q.wxyz).SE3() for q in self.traj.quats])
        trans_vec = SE3.Trans(self.p.T)
        self.frames = trans_vec * rots_vec  # make use of local transformations

        # derived data
        self.v = None
        self.acc = None
        self.om = None
        self.alp = None
        self._calc_derived_data()

        # initial conditions
        self.vec0 = self.vec_at(0)
        self.p0 = np.copy(self.p[:, 0]).reshape(3, 1)
        self.r0 = np.copy(self.r[:, 0]).reshape(3, 1)
        self.R0 = np.copy(self.R[0])
        self.q0: Quaternion = Quaternion(val=self.q[:, 0], do_normalise=True)
        self.v0 = np.copy(self.v[:, 0]).reshape(3, 1)
        self.om0 = np.copy(self.om[:, 0]).reshape(3, 1)
        self.acc0 = np.copy(self.acc[:, 0]).reshape(3, 1)
        self.alp0 = np.copy(self.alp[:, 0]).reshape(3, 1)

        # notch
        self.with_notch = with_notch
        self.is_rotated = False
        self.rotated = None

        self.notch = None
        self.notch_d = None
        self.notch_dd = None

        if self.flag_interpolated and self.traj.with_notch:
            self.notch = self.traj.notch
            self.notch_d = self.traj.notch_d
            self.notch_dd = self.traj.notch_dd

        if with_notch:
            # self._gen_notch_values()
            self._read_notch_from_traj()
            self.gen_rotated()

    @property
    def flag_interpolated(self):
        return self.traj.is_interpolated

    def update_sim_params(self, config):
        """Updates time-related info in the config, from camera data."""
        config.max_vals = self.max_vals
        config.min_t = self.min_t
        config.max_t = self.max_t
        config.total_data_pts = (self.max_vals - 1) * config.interframe_vals + 1

    def interpolate(self, interframe_vals):
        interp_traj = Interpolator(interframe_vals, self).interpolated
        interp_traj.with_notch = self.with_notch

        cam = CameraInterpolated(interp_traj)
        cam.with_notch = self.with_notch
        return cam

    def _calc_derived_data(self) -> None:
        if self.flag_interpolated:
            self.v = self.traj.v
            self.om = self.traj.om
            self.acc = self.traj.acc
            self.alp = self.traj.alp
        else:
            self.v = np.gradient(self.p, self.dt, axis=-1)
            self.acc = np.gradient(self.v, self.dt, axis=-1)

            ang_WC = np.asarray([q.euler_xyz_rad for q in self.traj.quats]).T
            self.om = np.gradient(ang_WC, self.dt, axis=-1)
            self.alp = np.gradient(self.om, self.dt, axis=-1)

    def gen_rotated(self):
        rotated_traj = VisualTraj("camera rot")
        rotated_traj.t = copy(self.t)
        rotated_traj.x = copy(self.traj.x)
        rotated_traj.y = copy(self.traj.y)
        rotated_traj.z = copy(self.traj.z)
        rotated_traj.quats = []

        for i, t in enumerate(self.t):
            real_quat = self.traj.quats[i]

            ang_notch = self.get_notch_vec_at(i)[0]
            notch_quat = Quaternion(val=np.array([0, 0, ang_notch]), euler="xyz")

            rotated_quat = notch_quat * real_quat

            rotated_traj.quats.append(rotated_quat)
            rotated_traj.qx.append(rotated_quat.x)
            rotated_traj.qy.append(rotated_quat.y)
            rotated_traj.qz.append(rotated_quat.z)
            rotated_traj.qw.append(rotated_quat.w)

        rotated_traj.is_rotated = True
        rotated_traj._gen_euler_angles()
        self.rotated = Camera(filepath=None, traj=rotated_traj)
        self.rotated.is_rotated = True
        self.rotated.with_notch = self.with_notch

        self.rotated.notch = self.notch
        self.rotated.notch_d = self.notch_d
        self.rotated.notch_dd = self.notch_dd

    def _read_notch_from_traj(self):
        assert self.with_notch is True
        self.notch = self.traj.notch
        self.notch_d = self.traj.notch_d
        self.notch_dd = self.traj.notch_dd

    def _gen_notch_values(self):
        def traj_gen(z0, zT, t, t_prev, T):
            t = t - t_prev
            T = T - t_prev

            z_n = z0 + (zT - z0) * (
                35 * (t / T) ** 4
                - 84 * (t / T) ** 5
                + 70 * (t / T) ** 6
                - 20 * (t / T) ** 7
            )

            z_n_d = (zT - z0) * (
                35 * 4 / (T ** 4) * t ** 3
                - 84 * 5 / (T ** 5) * t ** 4
                + 70 * 6 / (T ** 6) * t ** 5
                - 20 * 7 / (T ** 7) * t ** 6
            )
            z_n_dd = (zT - z0) * (
                35 * 4 * 3 / (T ** 4) * t ** 2
                - 84 * 5 * 4 / (T ** 5) * t ** 3
                + 70 * 6 * 5 / (T ** 6) * t ** 4
                - 20 * 7 * 6 / (T ** 7) * t ** 5
            )

            return z_n, z_n_d, z_n_dd

        ang_vals = np.pi * np.array([0, 0, 0.9, 0.9, -0.9 / 2, -0.9 / 2])

        ang_prev = ang_vals[0]
        t_part = self._gen_t_partition()
        t_prev = t_part[0][0]

        traj = [0] * len(t_part)
        traj_d = [0] * len(t_part)
        traj_dd = [0] * len(t_part)

        for i, ang_k in enumerate(ang_vals[1:]):
            t_max = t_part[i][-1]
            traj[i], traj_d[i], traj_dd[i] = traj_gen(
                ang_prev, ang_k, t_part[i], t_prev, t_max
            )
            t_prev = t_max
            ang_prev = ang_k

        self.notch = np.concatenate(traj).ravel()
        self.notch_d = np.concatenate(traj_d).ravel()
        self.notch_dd = np.concatenate(traj_dd).ravel()

        self.traj.notch = np.concatenate(traj).ravel()
        self.traj.notch_d = np.concatenate(traj_d).ravel()
        self.traj.notch_dd = np.concatenate(traj_dd).ravel()

        # self.traj.write_notch()
        # sys.exit()

    def _gen_t_partition(self):
        partitions = np.array([0, 0.1, 0.45, 0.5, 0.9, 1])
        t_part = [0] * (len(partitions) - 1)
        p_prev = 0
        for i, p in enumerate(partitions[1:]):
            p_k = int(np.ceil(p * self.max_vals))
            t_part[i] = np.array(self.t[p_prev:p_k])
            p_prev = p_k
        return t_part

    def get_notch_vec_at(self, i: int) -> np.ndarray:
        if self.with_notch:
            return np.array([self.notch[i], self.notch_d[i], self.notch_dd[i]])
        else:
            return np.array([0, 0, 0])

    def vec_at(self, i: int) -> List[np.ndarray]:
        p = np.copy(self.p[:, i]).reshape(3, 1)
        R = np.copy(self.R[i])
        v = np.copy(self.v[:, i]).reshape(3, 1)
        om = np.copy(self.om[:, i]).reshape(3, 1)
        acc = np.copy(self.acc[:, i]).reshape(3, 1)
        alp = np.copy(self.alp[:, i]).reshape(3, 1)

        return [p, R, v, om, acc, alp]

    def plot(self, config):
        CameraPlot(self, config).plot()

    def plot_notch(self, config):
        CameraPlot(self).plot_notch(config)


def _get_timestamp_index(timestamps: np.ndarray, max_t: float) -> float:
    """Get index where the timestamp <= max_t."""
    return max([i for i, t in enumerate(timestamps) if t <= max_t])


@dataclass(frozen=True)
class CameraMeasurementArray:
    t: np.ndarray

    p: np.ndarray
    R: List[np.ndarray]
    v: np.ndarray

    om: np.ndarray
    acc: np.ndarray
    alp: np.ndarray

    notch: np.ndarray
    notch_d: np.ndarray
    notch_dd: np.ndarray

    @staticmethod
    def from_camera(
        camera: Camera, old_t: float, new_t: float
    ) -> CameraMeasurementArray:
        """After old_t, up till new_t."""
        old_i = _get_timestamp_index(camera.t, old_t)
        new_i = _get_timestamp_index(camera.t, new_t)

        t = camera.t[old_i + 1 : new_i + 1]
        p = camera.p[:, old_i + 1 : new_i + 1]
        R = camera.R[old_i + 1 : new_i + 1]
        v = camera.v[:, old_i + 1 : new_i + 1]
        om = camera.om[:, old_i + 1 : new_i + 1]
        acc = camera.acc[:, old_i + 1 : new_i + 1]
        alp = camera.alp[:, old_i + 1 : new_i + 1]

        if camera.with_notch:
            notch = camera.notch[old_i + 1 : new_i + 1]
            notch_d = camera.notch_d[old_i + 1 : new_i + 1]
            notch_dd = camera.notch_dd[old_i + 1 : new_i + 1]
        else:
            notch = [0] * len(t)
            notch_d = [0] * len(t)
            notch_dd = [0] * len(t)

        return CameraMeasurementArray(
            t, p, R, v, om, acc, alp, notch, notch_d, notch_dd
        )

    def at_index(self, i: int) -> CameraMeasurementPoint:
        t = self.t[i]

        p = self.p[:, i]
        R = self.R[i]
        v = self.v[:, i]
        om = self.om[:, i]
        acc = self.acc[:, i]
        alp = self.alp[:, i]

        notch = self.notch[i]
        notch_d = self.notch_d[i]
        notch_dd = self.notch_dd[i]

        return CameraMeasurementPoint(
            t, p, R, v, om, acc, alp, notch, notch_d, notch_dd
        )


class CameraInterpolated(Camera):
    def __init__(self, traj: VisualTraj):
        assert traj.is_interpolated
        super().__init__(filepath=None, traj=traj)

    @property
    def interframe_vals(self):
        return self.traj.interframe_vals
