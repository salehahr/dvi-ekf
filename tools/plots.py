from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt

if TYPE_CHECKING:
    import numpy as np


def plot_imu(
    t: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    r: np.ndarray,
    omega: np.ndarray,
    p_recon: np.ndarray,
    v_recon: np.ndarray,
    eul_recon: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 5)
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(t, p[:, i])
        ax.plot(t, p_recon[:, i], ":")

        ax = axes[i, 1]
        ax.plot(t, v[:, i])
        ax.plot(t, v_recon[:, i], ":")

        ax = axes[i, 2]
        ax.plot(t, a[:, i])

        ax = axes[i, 3]
        ax.plot(t, r[:, i])
        ax.plot(t, eul_recon[:, i], ":")

        ax = axes[i, 4]
        ax.plot(t, omega[:, i])
    axes[0, 0].set_title("position $p$")
    axes[0, 1].set_title("velocity $v$")
    axes[0, 2].set_title("acceleration $a$")
    axes[0, 3].set_title("Euler angles $Z-Y-Z$")
    axes[0, 4].set_title("ang. velocity $\omega$")

    plt.suptitle("IMU")
    plt.show()
