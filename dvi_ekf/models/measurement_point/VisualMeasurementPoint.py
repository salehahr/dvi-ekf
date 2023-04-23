from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dvi_ekf.tools.Quaternion import Quaternion


@dataclass(frozen=True)
class VisualMeasurementPoint:
    """Visual measurements from SLAM
    containing position and orientation."""

    t: float

    x: float
    y: float
    z: float

    q: Quaternion

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def vec(self) -> np.ndarray:
        return np.hstack((self.pos, self.q.xyzw))
