import numpy as np

np.set_printoptions(suppress=True, precision=3)


def np_string(arr: np.ndarray, precision: int = 4) -> str:
    """Formats numpy arrays for printing."""
    arr = np.array(arr) if isinstance(arr, list) else arr
    return np.array2string(arr, precision=precision, suppress_small=True)
