import numpy as np


def to_world_coords(R_WB: np.ndarray, vec: np.ndarray) -> np.ndarray:
    if isinstance(R_WB, list):
        assert vec.shape[1] == 3
        return np.asarray([Rot @ vec[i, :] for i, Rot in enumerate(R_WB)])
    else:
        return R_WB @ vec
