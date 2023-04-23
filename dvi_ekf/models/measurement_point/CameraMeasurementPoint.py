from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CameraMeasurementPoint:
    t: float

    p: np.ndarray
    R: np.ndarray
    v: np.ndarray

    om: np.ndarray
    acc: np.ndarray
    alp: np.ndarray

    notch: float
    notch_d: float
    notch_dd: float

    @property
    def vec(self) -> List[np.ndarray]:
        return [self.p, self.R, self.v, self.om, self.acc, self.alp]

    @property
    def notch_arr(self) -> List[float]:
        return [self.notch, self.notch_d, self.notch_dd]
