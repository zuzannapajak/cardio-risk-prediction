# scripts/dim_red/utils.py
from typing import Union
import numpy as np
import pandas as pd

ArrayLike = Union[pd.DataFrame, pd.Series, np.ndarray]

def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert pandas objects or arrays to a NumPy array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)
