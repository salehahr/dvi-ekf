from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

from Filter.States import States

if TYPE_CHECKING:
    from Models.Camera import Camera
    from Models.Imu import Imu


def generate_imudof_ic(gt_dofs: np.ndarray) -> np.ndarray:
    """
    Generates synthetic initial conditions of the IMU dofs
    by perturbing the real DOFs

    e.g.
    DOFs (real) : [ 0.    0.  0.  0.  0. 20.]
    DOFs (IC)   : [ 0.05  0.  0.  3.  3. 23.]
    """
    # dofs0_rot_real = gt_dofs[:3]
    # dofs0_tr_real = gt_dofs[3:]
    #
    # # perturbations
    # delta_ang_rad = np.deg2rad(3)
    # delta_trans_cm = 3
    #
    # # # random perturbations
    # # delta_dof_rot = np.random.normal(loc=0, scale=delta_ang_rad, size=(3,))
    # # delta_dof_tr = np.random.normal(loc=0, scale=delta_trans_cm, size=(3,))
    #
    # # const. perturbations
    # delta_dof_rot = [delta_ang_rad, *dofs0_rot_real[1:]]
    # delta_dof_tr = [x + delta_trans_cm for x in dofs0_tr_real]
    #
    # return [*delta_dof_rot, *delta_dof_tr]

    # TODO: change this later; simulations currently use perfect initial conditions
    return gt_dofs


def get_upd_values(obj: object, comp: str, indices: Union[int, list]):
    """returns obj (e.g. self.traj or self.imu.ref) component
    at update timestamps only."""
    if indices == -1:
        return obj.__dict__[comp][-1]
    else:
        return np.array([obj.__dict__[comp][i] for i in indices])


def get_ic(camera: Camera, imu: Imu, imudof_ic: np.ndarray) -> States:
    """
    Returns the initial states given the camera and IMU states,
    as well as the initial values of the IMU DOFS.
    :param camera: camera object
    :param imu: IMU object
    :param imudof_ic: initial conditions of IMU degrees of freedom
    :return:
    """
    cam_reference = camera.rotated if camera.rotated else camera
    notch0 = cam_reference.get_notch_at(0)
    W_p_BW_0, R_WB_0, WW_v_BW_0 = imu.ref_vals(cam_reference.vec0, notch0)

    return States(
        p=W_p_BW_0,
        v=WW_v_BW_0,
        q=R_WB_0,
        dofs=imudof_ic,
        notch_dofs=notch0,
        p_cam=cam_reference.p0,
        q_cam=cam_reference.q0,
    )
