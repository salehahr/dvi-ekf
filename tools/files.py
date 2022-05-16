import os

import pandas as pd


def parse(filepath: str, max_vals: int) -> pd.DataFrame:
    """
    Extract data from file.
    :param filepath: filepath containing measurement data
    :param max_vals: maxinum number of data to store
    """
    _, extension = os.path.splitext(filepath)
    assert extension == ".csv"

    return pd.read_csv(filepath, nrows=max_vals)
