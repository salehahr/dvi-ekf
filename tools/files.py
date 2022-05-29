from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from Filter.Quaternion import Quaternion
from tools.spatial import solve_for_omega

if TYPE_CHECKING:
    from spatialmath import SE3


def parse(filepath: str, max_vals: int) -> pd.DataFrame:
    """
    Extract data from file.
    :param filepath: filepath containing measurement data
    :param max_vals: maxinum number of data to store
    """
    _, extension = os.path.splitext(filepath)
    assert extension == ".csv"

    return pd.read_csv(filepath, nrows=max_vals)


def write_angvel_data(
    filename: str, imu_frames_interp: SE3, t_interp: np.ndarray, dt: float
) -> None:
    with open(filename, "w") as f:
        title = ",".join(["t", "om_x", "om_y", "om_z"]) + "\n"
        f.write(title)

        for i in range(len(imu_frames_interp)):
            if i < len(imu_frames_interp) - 1:
                q0 = Quaternion(imu_frames_interp[i].R)
                q1 = Quaternion(imu_frames_interp[i + 1].R)
                omega = solve_for_omega(q0, q1, dt)
            else:
                q0 = Quaternion(imu_frames_interp[-2].R)
                q1 = Quaternion(imu_frames_interp[-1].R)
                omega = solve_for_omega(q0, q1, dt, eval_at_q0=False)
            data = [f"{t_interp[i]:.5f}", *[f"{x:.10f}" for x in omega]]
            data_str = ",".join(data) + "\n"
            f.write(data_str)


def save_tuned_params(x, filename=None):
    filename = "./opt-tune-params.txt" if not filename else filename
    with open(filename, "w+") as f:
        f.write(x)
