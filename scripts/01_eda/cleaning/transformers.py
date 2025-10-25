from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# ===== 1) TYPE & VALUE COERCION ==============================================
# =============================================================================

class ConvertToNumeric(BaseEstimator, TransformerMixin):
    """
    Convert selected columns to numeric dtype.

    - Non-convertible values are coerced to NaN (errors='coerce').
    - Safe for full-dataset cleaning (no learned parameters).
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            # Coerce non-numeric to NaN
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X


class InvalidValueToNaN(BaseEstimator, TransformerMixin):
    """
    Placeholder for generic invalid→NaN rules.

    Currently a no-op (returns X unchanged).
    Keep this in the cleaning stage to centralize any future generic rules,
    e.g., replace negative ages, special placeholders, or infinities.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Intentionally a no-op. Add generic invalid rules here if needed.
        return X


# =============================================================================
# ===== 2) RANGE / PLAUSIBILITY CHECKS ========================================
# =============================================================================

class BoundsToNaN(BaseEstimator, TransformerMixin):
    """
    Set physiologically invalid numeric values to NaN using per-column bounds.

    Parameters
    ----------
    bounds : dict[str, tuple[float, float]]
        Mapping of column -> (lower_inclusive, upper_inclusive)
        Use np.inf for open-ended upper bounds.

    Example
    -------
    bounds = {
        "age": (18, 120),
        "serum_sodium": (110, 160),
        "time": (0, np.inf),
    }
    """
    def __init__(self, bounds: Dict[str, tuple[float, float]]):
        self.bounds = bounds

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds.items():
            if col in X.columns:
                X.loc[(X[col] < lo) | (X[col] > hi), col] = np.nan
        return X


# =============================================================================
# ===== 3) BINARY INTEGRITY ====================================================
# =============================================================================

class EnsureBinaryInt(BaseEstimator, TransformerMixin):
    """
    Coerce binary columns to {0, 1} integers.

    Behavior:
    - Coerce to numeric (non-numeric -> NaN)
    - Round values (e.g., 0.0/1.0 -> 0/1)
    - Clip to [0, 1]
    - Cast to Python int (via pandas nullable Int64 to preserve NaN during ops)
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].round().clip(0, 1).astype("Int64").astype(int)
        return X


# =============================================================================
# ===== 4) COLUMN MANIPULATION (DROP / ROUND/CAST) ============================
# =============================================================================

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Drop a list of columns if present.

    Note:
    - This class duplicates DropColumnTransformer below (kept for backward compatibility).
      Prefer using `DropColumnTransformer` for consistency.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors="ignore")


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from the DataFrame.

    Useful for excluding features that should not be passed
    to downstream models or transformations.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        existing = [c for c in self.columns if c in X.columns]
        return X.drop(columns=existing, errors="ignore")


class RoundAndCastInt(BaseEstimator, TransformerMixin):
    """
    Round numeric columns and cast to int64.

    Caution:
    - Rounding discards fractional information; use ONLY if integers are semantically required.
    - This transformer is deterministic and safe for cleaning.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # Round then cast to int64; errors='ignore' leaves non-castable dtypes unchanged
                X[col] = X[col].round().astype("int64", errors="ignore")
        return X


class DropHighNullColumns(BaseEstimator, TransformerMixin):
    """
    Drop columns whose missing-rate exceeds a given threshold.

    Parameters
    ----------
    threshold : float, default=0.4
        Proportion of missing values above which a column is dropped.
    """
    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold
        self.columns_to_drop_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop_, errors="ignore")


# =============================================================================
# ===== 5) CATEGORICAL IMPUTATION =============================================
# =============================================================================

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in categorical columns.

    Parameters
    ----------
    strategy : {'mode', 'constant'}, default='mode'
        - 'mode': fill with column mode (most frequent)
        - 'constant': fill with `fill_value`
    fill_value : str, default='Missing'
        Used when strategy='constant'.

    Notes
    -----
    - Operates on non-numeric dtypes only.
    - Safe for cleaning, though 'mode' is technically data-derived.
      If you want to avoid any training-data dependence, prefer
      `CategoricalMissingCategoryImputer` with a fixed placeholder.
    """
    def __init__(self, strategy: str = "mode", fill_value: str = "Missing"):
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_values_: Dict[str, str] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.columns_ = X.select_dtypes(exclude=np.number).columns.tolist()

        for col in self.columns_:
            if self.strategy == "mode":
                mode_series = X[col].mode()
                self.fill_values_[col] = mode_series[0] if not mode_series.empty else self.fill_value
            elif self.strategy == "constant":
                self.fill_values_[col] = self.fill_value
            else:
                raise ValueError("strategy must be 'mode' or 'constant'")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_values_[col])
        return X


class CategoricalMissingCategoryImputer(BaseEstimator, TransformerMixin):
    """
    Replace missing values in categorical columns with a fixed placeholder category.

    Parameters
    ----------
    fill_value : str, default='Missing'
        Placeholder category to insert for NA values.

    Notes
    -----
    - Operates on non-numeric dtypes only.
    - Fully deterministic; recommended for the cleaning stage if you
      want to avoid any training-data dependence.
    """
    def __init__(self, fill_value: str = "Missing"):
        self.fill_value = fill_value
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = X.select_dtypes(exclude=np.number).columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_value)
        return X
