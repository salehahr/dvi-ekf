from __future__ import annotations

import math
import os
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import yaml
from pydantic import BaseModel, validator

from tools import utils

if TYPE_CHECKING:
    from Models.Probe import Probe

np.set_printoptions(suppress=True, precision=3)


def np_string(arr: np.ndarray, precision: int = 4) -> str:
    """Formats numpy arrays for printing."""
    arr = np.array(arr) if isinstance(arr, list) else arr
    return np.array2string(arr, precision=precision, suppress_small=True)


class SimMode(Enum):
    """Simulation categories: either run the filter or tune the filter."""

    RUN = "run"
    TUNE = "tune"


class SimConfig(BaseModel):
    """Configuration of the simulation."""

    # mode: either run or tune
    mode: SimMode

    # filepath configurations
    data_folder: str
    traj_name: str
    notch_traj_name: str

    # number of times to run the simulation (e.g. for averaging across runs)
    num_kf_runs: int = 1

    # sensor setup DOFs to freeze
    frozen_dofs: List[int]

    # boolean options
    do_plot: bool
    do_fast_sim: bool

    @validator("mode")
    def set_mode(cls, v) -> SimMode:
        return SimMode(v)


class CameraNoise(BaseModel):
    """
    Settings for the noise matrix of the visual measurements.
    Angle values are given in degrees by the user in the yaml file,
    but are converted to radians for simulation.
    """

    position: List[float]
    theta: List[float]
    notch: float

    @validator("theta", "notch")
    def to_radians(cls, v) -> Union[list, float]:
        return np.deg2rad(v).tolist() if isinstance(v, list) else np.deg2rad(v)

    @property
    def vec(self) -> np.ndarray:
        return np.hstack((self.position, self.theta, self.notch))


class CameraConfig(BaseModel):
    """
    Configuration of the camera trajectory.
    """

    start_frame: Optional[int]
    total_frames: Optional[Union[str, int]]
    scale: int
    with_notch: bool = False
    noise: CameraNoise

    @validator("total_frames")
    def all_frames(cls, v) -> Optional[int]:
        return None if v == "all" else v


class ImuConfig(BaseModel):
    """
    Configuration of the IMU trajectory.
    """

    interframe_vals: int
    noise_sample_rate: float
    gravity: float

    @property
    def noise_gyro(self) -> float:
        """in deg/s."""
        return 0.005 * math.sqrt(self.noise_sample_rate)

    @property
    def stdev_omega(self) -> List[float]:
        """in rad/s."""
        return [np.deg2rad(self.noise_gyro)] * 3

    @property
    def stdev_accel(self) -> List[float]:
        """in cm/s^2"""
        return [400 * 1e-6 * self.gravity * math.sqrt(self.noise_sample_rate)] * 3


class ModelConfig(BaseModel):
    """
    Dimensions of the sensor setup model.
    """

    length: float
    angle: float

    @validator("angle")
    def to_radians(cls, v) -> float:
        return np.deg2rad(v)


class States(BaseModel):
    """
    Values of the filter states.
    """

    imu_pos: List[float]
    imu_vel: List[float]
    imu_theta: List[float]
    dofs_rot: List[float]
    dofs_trans: List[float]
    notch: List[float]
    camera_pos: List[float]
    camera_theta: List[float]

    @validator("imu_theta", "dofs_rot", "notch", "camera_theta")
    def to_radians(cls, v) -> List[float]:
        return np.deg2rad(v).tolist()

    @property
    def vec(self) -> np.ndarray:
        return np.hstack(
            (
                self.imu_pos,
                self.imu_vel,
                self.imu_theta,
                self.dofs_rot,
                self.dofs_trans,
                self.notch,
                self.camera_pos,
                self.camera_theta,
            )
        )


class InitialConditions(BaseModel):
    """
    Initial values of the filter states.
    """

    cov0: States
    x0: States

    @property
    def cov0_matrix(self) -> np.ndarray:
        return np.square(np.diag(self.cov0.vec))


class DofsNoise(BaseModel):
    """
    Settings for the process matrix (DOFs random walk parameters).
    """

    translation: List[float]
    rotation: List[float]
    notch_accel: float

    @validator("rotation", "notch_accel")
    def to_radians(cls, v) -> Union[list, float]:
        return np.deg2rad(v).tolist() if isinstance(v, list) else np.deg2rad(v)

    @property
    def vec(self) -> np.ndarray:
        return np.hstack((self.rotation, self.translation, self.notch_accel))


class FilterConfig(BaseModel):
    """
    Configuration of the filter.
    """

    ic: InitialConditions
    process_noise: Dict

    @property
    def noise_dofs(self) -> DofsNoise:
        return DofsNoise(**self.process_noise["dofs"])


class Config(object):
    """
    Configuration object that contains settings for the
    simulation, models, trajectories as well as initial conditions.
    """

    def __init__(self, filepath: str):
        """
        :param filepath: yaml file
        """

        with open(filepath) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.sim = SimConfig(**config["simulation"])
        self.camera = CameraConfig(**config["camera"])
        self.imu = ImuConfig(**config["imu"])
        self.model = ModelConfig(**config["model"])
        self.filter = FilterConfig(**config["filter"])

        # simulation params
        self.interframe_vals = self.imu.interframe_vals
        self.cov0_matrix = self.filter.ic.cov0_matrix
        self.with_notch = self.camera.with_notch

        # these get initialised after loading the Camera object
        self.max_vals = self.camera.total_frames
        self.min_t: Optional[float] = None
        self.max_t: Optional[float] = None
        self.total_data_pts: Optional[int] = None

        # overwrite some attributes if fast_sim is set
        if self.sim.do_fast_sim:
            self.interframe_vals = 1
            self.max_vals = 10

        # probe
        self.frozen_dofs: List = self.sim.frozen_dofs
        self.gt_joint_dofs: Optional[List] = None
        self.gt_imu_dofs: Optional[np.ndarray] = None
        self.ic_imu_dofs: Optional[np.ndarray] = None
        self._dofs_updated = False

        # noise
        self.process_noise_rw_std = self.filter.noise_dofs.vec / self.interframe_vals
        self.process_noise_rw_var = np.square(self.process_noise_rw_std)
        self.meas_noise_std = self.camera.noise.vec
        self.meas_noise_var = np.square(self.meas_noise_std)

        # save and plot
        data_path = self.sim.data_folder
        self.img_path = "./docs/img"
        self.traj_path = os.path.join(data_path, "trajs")
        self.traj_name: str = self.sim.traj_name

        self.mse: Optional[float] = None
        self.do_plot: bool = self.sim.do_plot

    @property
    def dofs_updated(self) -> bool:
        return self._dofs_updated

    def update_dofs(self, probe: Probe) -> None:
        self.gt_joint_dofs = probe.joint_dofs.copy()
        self.gt_imu_dofs = probe.imu_dofs.copy()
        self.ic_imu_dofs = utils.generate_imudof_ic(self.gt_imu_dofs)

        self._dofs_updated = True

    def print(self) -> None:
        cov0 = self.filter.ic.cov0
        print(
            "Configuration:",
            f"\t Trajectory          : {self.traj_name}",
            f"\t Mode                : {self.sim.mode.value}",
            "",
            f"\t Num. cam. frames    : {self.max_vals}",
            f"\t Num. IMU data       : {self.total_data_pts}",
            f"\t(num. IMU b/w frames : {self.interframe_vals})",
            "",
            f"\t Frozen DOFs         : {self.frozen_dofs}",
            "",
            f"\t ## Noise values",
            f"\t #  P0: Initial process noise",
            f"\t std_dp             = {cov0.imu_pos[0]:.1f} \t cm",
            f"\t std_dv             = {cov0.imu_vel[0]:.1f} \t cm/s",
            f"\t std_dtheta         = {np.rad2deg(cov0.imu_theta[0]):.1f} \t deg",
            f"\t std_ddofs_rot      = {np.rad2deg(cov0.dofs_rot[0]):.1f} \t deg",
            f"\t std_ddofs_trans    = {cov0.dofs_trans[0]:.1f} \t cm",
            f"\t std_dnotch         = {np.rad2deg(cov0.notch[0]):.1f} \t deg",
            f"\t std_dnotchd        = {np.rad2deg(cov0.notch[1]):.1f} \t deg/s",
            f"\t std_dnotchdd       = {np.rad2deg(cov0.notch[2]):.1f} \t deg/s^2",
            f"\t std_dp_cam         = {cov0.camera_pos[0]:.1f} \t cm",
            f"\t std_dtheta_cam     = {cov0.camera_theta[0]:.1f} \t deg",
            "",
            f"\t #  Q: IMU measurement noise",
            f"\t std_acc            = {np_string(self.imu.stdev_accel)} cm/s^2",
            f"\t std_om             = {np_string(np.rad2deg(self.imu.stdev_omega))} deg/s",
            "",
            f"\t #  Q: IMU dofs random walk noise",
            f"\t std_dofs_r  = {np_string(np.rad2deg(self.process_noise_rw_std[0:3]))} deg",
            f"\t std_dofs_p  = {np_string(self.process_noise_rw_std[3:6])} cm",
            "",
            f"\t #  Q: Notch accel random walk noise",
            f"\t std_notchdd = {np.rad2deg(self.process_noise_rw_std[6]):.4f} deg/s^2",
            "",
            f"\t #  R: Measurement noise",
            f"\t std_pc     = {np_string(self.meas_noise_std[0:3])} cm",
            f"\t std_qc     = "
            + f"{np_string(np.rad2deg(self.meas_noise_std[3:6]))} deg",
            f"\t std_notch  = "
            + f"{np_string(np.rad2deg(self.meas_noise_std[6]))} deg",
            "",
            sep="\n",
        )
        self._print_dofs()

    def _print_dofs(self) -> None:
        if self.gt_imu_dofs:
            print(f"DOFs (real) : {np_string(self.gt_imu_dofs)}")

        if self.ic_imu_dofs:
            print(f"DOFs (IC)   : {np_string(self.ic_imu_dofs)}\n")
