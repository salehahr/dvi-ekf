from __future__ import annotations

import os
from abc import ABC
from typing import TYPE_CHECKING, List, Optional

from dvi_ekf.tools import files

if TYPE_CHECKING:
    import numpy as np


class TrajectoryBase(ABC):
    """
    Base trajectory class which requires a trajectory name, trajectory labels, and a filepath.
    """

    labels: List[str] = ["t"]

    def __init__(
        self,
        name: str,
        filepath: Optional[str] = None,
        cap: Optional[int] = None,
        interframe_vals: int = 0,
    ):
        self.name = name
        self.filepath = filepath

        # Number of interpolated values between frames (F0 < interpolated_data <= F1)
        self.interframe_vals = interframe_vals
        self.is_interpolated = False

        self.t: Optional[np.ndarray] = None

        self.reset()
        self.__parse_from_file(filepath, cap)

    def reset(self):
        """Reinitialises data labels."""
        for label in self.labels:
            setattr(self, label, None)

    @property
    def num_values(self) -> int:
        """Number of values stored."""
        return len(self.t)

    def __parse_from_file(self, filepath: str, cap: int) -> None:
        """
        Parse trajectory from file if file is given and file exists.

        :param filepath: trajectory filepath
        :param cap: maximum number of measurement values
        :return:
        """
        if self.filepath is None:
            return

        if os.path.exists(self.filepath):
            self.__fill_attributes(cap)
        else:
            files.handle_not_exist(filepath)

    def __fill_attributes(self, max_vals: int) -> None:
        """
        Extract data from file.
        :param max_vals: maxinum number of data to store
        """
        dataframe = files.get_dataframe(self.filepath, max_vals)
        for label in self.labels:
            data = dataframe[label].to_numpy()
            setattr(self, label, data)
