import math
import os
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from pydantic import BaseModel, validator

from tools import utils

np.set_printoptions(suppress=True, precision=3)


def np_string(arr):
    """For formatting numpy arrays when printing."""
    arr = np.array(arr) if isinstance(arr, list) else arr
    return np.array2string(arr, precision=4, suppress_small=True)


class SimMode(Enum):
    RUN = "run"
    TUNE = "tune"


class SimConfig(BaseModel):
    mode: SimMode
    data_folder: str
    traj_name: str
    num_kf_runs: int
    frozen_dofs: List[int]
    do_plot: bool
    do_fast_sim: bool

    @validator("mode")
    def to_radians(cls, v) -> SimMode:
        return SimMode(v)


class CameraNoise(BaseModel):
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
    start_frame: Optional[int]
    total_frames: Optional[Union[str, int]]
    scale: int
    noise: CameraNoise

    @validator("total_frames")
    def all_frames(cls, v) -> Optional[int]:
        return None if v == "all" else v


class ImuConfig(BaseModel):
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
    length: float
    angle: float

    @validator("angle")
    def to_radians(cls, v) -> float:
        return np.deg2rad(v)


class States(BaseModel):
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
    cov0: States
    x0: States

    @property
    def cov0_matrix(self) -> np.ndarray:
        return np.square(np.diag(self.cov0.vec))


class DofsNoise(BaseModel):
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
    ic: InitialConditions
    process_noise: Dict

    @property
    def noise_dofs(self) -> DofsNoise:
        return DofsNoise(**self.process_noise["dofs"])


class Config(object):
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
        self.interframe_vals = 1 if self.sim.do_fast_sim else self.imu.interframe_vals
        self.cov0_matrix = self.filter.ic.cov0_matrix

        # these get initialised after loading the Camera object
        self.max_vals = 10 if self.sim.do_fast_sim else self.camera.total_frames
        self.min_t: Optional[float] = None
        self.max_t: Optional[float] = None
        self.total_data_pts: Optional[int] = None

        # probe
        self.frozen_dofs: List = self.sim.frozen_dofs
        self.gt_joint_dofs: Optional[List] = None
        self.gt_imu_dofs: Optional[List] = None
        self.ic_imu_dofs: Optional[List] = None

        # noise
        self.process_noise_rw_std = self.filter.noise_dofs.vec / self.interframe_vals
        self.process_noise_rw_var = np.square(self.process_noise_rw_std)
        self.meas_noise_std = self.camera.noise.vec
        self.meas_noise_var = np.square(self.meas_noise_std)

        # save and plot
        data_path = self.sim.data_folder
        self.img_path = "./doc/img"
        self.traj_path = os.path.join(data_path, "trajs")
        self.traj_name: str = self.sim.traj_name

        self.mse: Optional[float] = None
        self.do_plot: bool = self.sim.do_plot

    def update_dofs(self, probe) -> None:
        self.gt_joint_dofs = probe.joint_dofs.copy()
        self.gt_imu_dofs = probe.imu_dofs.copy()
        self.ic_imu_dofs = utils.generate_imudof_ic(self.gt_imu_dofs)

    def print(self) -> None:
        cov0 = self.filter.ic.cov0
        print(
            "Configuration: \n",
            f"\t Trajectory          : {self.traj_name}\n",
            f"\t Mode                : {self.sim.mode}\n\n",
            f"\t Num. cam. frames    : {self.max_vals}\n",
            f"\t Num. IMU data       : {self.total_data_pts}\n",
            f"\t(num. IMU b/w frames : {self.interframe_vals})\n\n",
            f"\t Frozen DOFs         : {self.frozen_dofs}\n\n",
            f"\t ## Noise values\n",
            f"\t #  P0: Initial process noise\n",
            f"\t std_dp             = {cov0.imu_pos[0]:.1f} \t cm\n",
            f"\t std_dv             = {cov0.imu_vel[0]:.1f} \t cm/s\n",
            f"\t std_dtheta         = {np.rad2deg(cov0.imu_theta[0]):.1f} \t deg\n",
            f"\t std_ddofs_rot      = {np.rad2deg(cov0.dofs_rot[0]):.1f} \t deg\n",
            f"\t std_ddofs_trans    = {cov0.dofs_trans[0]:.1f} \t cm\n",
            f"\t std_dnotch         = {np.rad2deg(cov0.notch[0]):.1f} \t deg\n",
            f"\t std_dnotchd        = {np.rad2deg(cov0.notch[1]):.1f} \t deg/s\n",
            f"\t std_dnotchdd       = {np.rad2deg(cov0.notch[2]):.1f} \t deg/s^2\n",
            f"\t std_dp_cam         = {cov0.camera_pos[0]:.1f} \t cm\n",
            f"\t std_dtheta_cam     = {cov0.camera_theta[0]:.1f} \t deg\n\n",
            f"\t #  Q: IMU measurement noise\n",
            f"\t std_acc            = {np_string(self.imu.stdev_accel)} cm/s^2\n",
            f"\t std_om             = {np_string(np.rad2deg(self.imu.stdev_omega))} deg/s\n\n",
            f"\t #  Q: IMU dofs random walk noise\n",
            f"\t std_dofs_r  = {np_string(np.rad2deg(self.process_noise_rw_std[0:3]))} deg\n",
            f"\t std_dofs_p  = {np_string(self.process_noise_rw_std[3:6])} cm\n\n",
            f"\t #  Q: Notch accel random walk noise\n",
            f"\t std_notchdd = {np.rad2deg(self.process_noise_rw_std[6]):.4f} deg/s^2\n\n",
            f"\t #  R: Measurement noise\n",
            f"\t std_pc     = {np_string(self.meas_noise_std[0:3])} cm \n",
            f"\t std_qc     = "
            + f"{np_string(np.rad2deg(self.meas_noise_std[3:6]))} deg\n",
            f"\t std_notch  = "
            + f"{np_string(np.rad2deg(self.meas_noise_std[6]))} deg\n\n",
        )
        self.print_dofs()

    def print_dofs(self) -> None:
        print(f"DOFs (real) : {np_string(self.gt_imu_dofs)}")
        print(f"DOFs (IC)   : {np_string(self.ic_imu_dofs)}\n")
