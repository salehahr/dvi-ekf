from enum import Enum


class SimMode(Enum):
    """Simulation categories: either run the filter or tune the filter."""

    RUN = "run"
    TUNE = "tune"
