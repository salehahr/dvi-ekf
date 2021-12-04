import math
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import yaml
from pydantic import BaseModel, validator

import tools
from Filter import States
from Models import Camera, Imu, SimpleProbe, SymProbe

np.set_printoptions(suppress=True, precision=3)


def np_string(arr):
    """For formatting np arrays when printing."""
    arr = np.array(arr) if isinstance(arr, list) else arr
    return np.array2string(arr, precision=4, suppress_small=True)


class SimMode(Enum):
    RUN = "run"
    TUNE = "tune"


class SimConfig(BaseModel):
    data_base_path: str
    traj_name: str
    num_kf_runs: int
    mode: SimMode
    do_plot: bool
    do_fast_sim: bool


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


class StatesConfig(BaseModel):
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


class InitialConditions(BaseModel):
    cov0: StatesConfig
    x0: StatesConfig

    @property
    def cov0_vector(self) -> np.ndarray:
        return np.hstack(
            (
                self.cov0.imu_pos,
                self.cov0.imu_vel,
                self.cov0.imu_theta,
                self.cov0.dofs_rot,
                self.cov0.dofs_trans,
                self.cov0.notch,
                self.cov0.camera_pos,
                self.cov0.camera_theta,
            )
        )

    @property
    def cov0_matrix(self) -> np.ndarray:
        return np.square(np.diag(self.cov0_vector))


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
    def __init__(self):
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        filter_config = FilterConfig(**config["filter"])

        self.camera = CameraConfig(**config["camera"])
        self.imu = ImuConfig(**config["imu"])
        model_config = ModelConfig(**config["model"])

        # simulation params
        self.sim = SimConfig(**config["simulation"])
        self.max_vals = self.camera.total_frames
        self.num_interframe_vals = (
            1 if self.sim.do_fast_sim else self.imu.interframe_vals
        )
        self.ic = filter_config.ic

        """ these get initialised after loading the Camera obj.
            ._gen_sim_params_from_cam() """
        self.min_t = None
        self.max_t = None
        self.total_data_pts = None

        # probe
        """ Container for probe object containing only the symbolic
            relative kinematics. """
        freeze = [1, 1, 1, 1, 1, 1]
        self.frozen_dofs = freeze
        probe = SimpleProbe(
            scope_length=model_config.length, theta_cam=model_config.angle
        )
        self.probe = probe
        self.sym_probe = SymProbe(probe)
        self.real_joint_dofs = probe.joint_dofs.copy()
        self.gt_imudof = probe.imu_dofs.copy()
        self.est_imudof_ic = tools.generate_imudof_ic(self.gt_imudof)

        # noise
        # # process (random walk components only)
        self.process_noise_rw_std = (
            filter_config.noise_dofs.vec / self.num_interframe_vals
        )

        # # measurement
        self.meas_noise_std = self.camera.noise.vec

        # saving
        self.mse = None
        self.saved_configs = [
            "meas_noise",
            "max_vals",
            "num_interframe_vals",
            "min_t",
            "max_t",
            "total_data_pts",
            "traj_name",
            "mse",
        ]

        # plot variables
        self.do_plot = self.sim.do_plot
        self.traj_name = self.sim.traj_name
        self.data_path = self.sim.data_base_path

        self.img_filepath_imu = "doc/img/kf_" + self.traj_name + "_imu.png"
        self.img_filepath_cam = "doc/img/kf_" + self.traj_name + "_cam.png"
        self.img_filepath_compact = "doc/img/kf_" + self.traj_name + "_compact.png"
        self.traj_kf_filepath = "data/trajs/kf_best_" + self.traj_name + ".txt"
        self.traj_imuref_filepath = "data/trajs/imu_ref_" + self.traj_name + ".txt"

    # auto square the covariances
    @property
    def process_noise_rw(self):
        return np.square(self.process_noise_rw_std)

    @property
    def meas_noise(self):
        return np.square(self.meas_noise_std)

    def get_camera(self):
        filepath_cam = f"./trajs/{self.traj_name}.txt"

        if self.max_vals:
            with_notch = True if self.max_vals > 10 else False
        else:
            with_notch = True
        # with_notch = False

        cam = Camera(
            filepath=filepath_cam,
            max_vals=self.max_vals,
            scale=self.camera.scale,
            with_notch=with_notch,
            start_at=self.camera.start_frame,
        )

        if with_notch:
            assert cam.rotated is not None
        else:
            assert cam.rotated is None

        self._gen_sim_params_from_cam(cam)

        return cam

    def _gen_sim_params_from_cam(self, camera):
        """Updates time-related info from camera data."""
        self.max_vals = camera.max_vals
        self.min_t = camera.min_t
        self.max_t = camera.max_t
        self.total_data_pts = (self.max_vals - 1) * self.num_interframe_vals + 1

    def get_imu(self, camera=None, gen_ref=False):
        """Generates IMU object from interpolated camera data."""
        cam_reference = camera.rotated if camera.rotated else camera
        camera_interp = cam_reference.interpolate(self.num_interframe_vals)
        return Imu(
            self.probe,
            camera_interp,
            self.imu.stdev_accel,
            self.imu.stdev_omega,
            gen_ref=gen_ref,
        )

    def get_ic(self, camera, imu):
        """Perfect initial conditions except for DOFs."""
        cam_reference = camera.rotated if camera.rotated else camera
        notch0 = cam_reference.get_notch_at(0)
        W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(cam_reference.vec0, notch0)

        x0 = States(
            W_p_BW_0,
            WW_v_BW_0,
            R_WB_0,
            self.est_imudof_ic,
            notch0,
            cam_reference.p0,
            cam_reference.q0,
        )

        return x0, self.ic.cov0_matrix

    def save(self, filename):
        """Saves come config parameters to text file."""
        configs = "\n".join([str(self.__dict__[s]) for s in self.saved_configs])

        with open(filename, "w+") as f:
            f.write(configs)

    def print_config(self):
        print(
            "Configuration: \n",
            f"\t Trajectory          : {self.traj_name}\n",
            f"\t Mode                : {self.sim.mode}\n\n",
            f"\t Num. cam. frames    : {self.max_vals}\n",
            f"\t Num. IMU data       : {self.total_data_pts}\n",
            f"\t(num. IMU b/w frames : {self.num_interframe_vals})\n\n",
            f"\t Frozen DOFs         : {self.frozen_dofs}\n\n",
            f"\t ## Noise values\n",
            f"\t #  P0: Initial process noise\n",
            f"\t std_dp             = {self.ic.cov0.imu_pos[0]:.1f} \t cm\n",
            f"\t std_dv             = {self.ic.cov0.imu_vel[0]:.1f} \t cm/s\n",
            f"\t std_dtheta         = {np.rad2deg(self.ic.cov0.imu_theta[0]):.1f} \t deg\n",
            f"\t std_ddofs_rot      = {np.rad2deg(self.ic.cov0.dofs_rot[0]):.1f} \t deg\n",
            f"\t std_ddofs_trans    = {self.ic.cov0.dofs_trans[0]:.1f} \t cm\n",
            f"\t std_dnotch         = {np.rad2deg(self.ic.cov0.notch[0]):.1f} \t deg\n",
            f"\t std_dnotchd        = {np.rad2deg(self.ic.cov0.notch[1]):.1f} \t deg/s\n",
            f"\t std_dnotchdd       = {np.rad2deg(self.ic.cov0.notch[2]):.1f} \t deg/s^2\n",
            f"\t std_dp_cam         = {self.ic.cov0.camera_pos[0]:.1f} \t cm\n",
            f"\t std_dtheta_cam     = {self.ic.cov0.camera_theta[0]:.1f} \t deg\n\n",
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

    def print_dofs(self):
        print(f"DOFs (real) : {np_string(self.gt_imudof)}")
        print(f"DOFs (IC)   : {np_string(self.est_imudof_ic)}\n")
