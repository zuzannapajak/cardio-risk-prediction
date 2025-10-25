from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from inspect import signature

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder as _SkOrdinalEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


# =============================================================================
# ===== 1) LABEL ENCODERS =====================================================
# =============================================================================

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Label-encode specific categorical columns with stable, frozen mappings.

    Notes
    -----
    - Mapping is learned on `fit` from TRAIN data (string-cast + '<MISSING>' for NaN).
    - Unseen categories at transform time -> -1.
    - Produces integer-typed columns suitable for tree models or as inputs to further steps.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.mappings_: Dict[str, Dict[str, int]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for col in self.columns:
            cats = pd.Series(X[col].astype("string")).fillna("<MISSING>").unique()
            mapping = {cat: i for i, cat in enumerate(sorted(cats))}
            self.mappings_[col] = mapping
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "mappings_")
        X = X.copy()
        for col in self.columns:
            if col not in X.columns:
                continue
            mapping = self.mappings_[col]
            as_str = X[col].astype("string").fillna("<MISSING>")
            X[col] = as_str.map(mapping).fillna(-1).astype(int)
        return X


class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Ordinal-encode columns using an explicit order for each feature.

    Parameters
    ----------
    categories_map : dict[str, list[Any]]
        Dict of column -> ordered category list.
        Unknown categories -> -1.

    Notes
    -----
    - Uses sklearn's OrdinalEncoder with unknown_value=-1.
    - Casts outputs to int where feasible (keeps -1).
    """
    def __init__(self, categories_map: Dict[str, List[Any]]):
        self.categories_map = categories_map
        self._encoder = None
        self.columns_: List[str] = list(categories_map.keys())

    def fit(self, X: pd.DataFrame, y=None):
        cats_in_order = [self.categories_map[c] for c in self.columns_]
        self._encoder = _SkOrdinalEncoder(
            categories=cats_in_order,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self._encoder.fit(X[self.columns_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self._encoder)
        X = X.copy()
        X[self.columns_] = self._encoder.transform(X[self.columns_])
        # cast to int where possible (unknowns are -1)
        for c in self.columns_:
            try:
                X[c] = X[c].astype(int)
            except Exception:
                pass
        return X


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    One-hot encode specified categorical columns.

    Parameters
    ----------
    columns : list[str]
        Columns to one-hot encode.
    sparse : bool, default=False
        Whether to output a scipy sparse matrix internally (converted to dense DataFrame if needed).
    drop_original : bool, default=True
        If True, original categorical columns are dropped.

    Notes
    -----
    - Compatible with sklearn >=1.2 (uses 'sparse_output' if available, else 'sparse').
    - Unknown categories are ignored during transform.
    """
    def __init__(self, columns, sparse: bool = False, drop_original: bool = True):
        self.columns = columns
        self.sparse = sparse
        self.drop_original = drop_original
        self._ohe = None
        self._feature_names = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        # Build encoder with version-appropriate arg
        params = signature(OneHotEncoder.__init__).parameters
        if "sparse_output" in params:
            self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=self.sparse)
        else:
            self._ohe = OneHotEncoder(handle_unknown="ignore", sparse=self.sparse)
        self._ohe.fit(X[self.columns])
        self._feature_names = self._ohe.get_feature_names_out(self.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        oh = self._ohe.transform(X[self.columns])
        if hasattr(oh, "toarray"):
            oh = oh.toarray()
        oh_df = pd.DataFrame(oh, columns=self._feature_names, index=X.index)

        if self.drop_original:
            X = X.drop(columns=[c for c in self.columns if c in X.columns])

        # ensure numeric dummies
        oh_df = oh_df.astype(np.float64)
        return pd.concat([X, oh_df], axis=1)


class FixedMappingEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns using fixed, explicit mappings.

    Parameters
    ----------
    mapping : dict[str, dict[Any, int]]
        Column -> category->code mapping.
    fallback_value : int, default=-1
        Value for unmapped / unseen categories.
    """
    def __init__(self, mapping: Dict[str, Dict[Any, int]], fallback_value: int = -1):
        self.mapping = mapping
        self.fallback_value = fallback_value
        self.columns_: List[str] = list(mapping.keys())

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, col_map in self.mapping.items():
            if col not in X.columns:
                continue
            X[col] = (
                X[col]
                .astype("string")
                .map(col_map)
                .fillna(self.fallback_value)
                .astype(int)
            )
        return X


# =============================================================================
# ===== 2) NORMALIZATION / SCALING ============================================
# =============================================================================

class UnitNormalizationTransformer(BaseEstimator, TransformerMixin):
    """
    Apply simple unit conversions / scaling factors per column.

    Example
    -------
    factors = {'age_days': 1/365}  # convert days to years
    """
    def __init__(self, factors: Dict[str, float]):
        self.factors = factors
        self.columns_: List[str] = list(factors.keys())

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, factor in self.factors.items():
            if col in X.columns:
                X[col] = X[col].astype(float) * float(factor)
        return X


class RobustScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Robust scaling per column: (x - median) / IQR, where IQR = Q3 - Q1.

    Notes
    -----
    - Falls back to eps if IQR ~ 0 to avoid division-by-zero.
    - Stats (median/IQR) are learned on TRAIN.
    """
    def __init__(self, columns: List[str], eps: float = 1e-12):
        self.columns = columns
        self.eps = eps
        self.stats_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.stats_.clear()
        for col in self.columns:
            if col not in X.columns:
                continue
            s = pd.to_numeric(X[col], errors="coerce")
            q1 = float(np.nanpercentile(s, 25))
            q3 = float(np.nanpercentile(s, 75))
            med = float(np.nanmedian(s))
            iqr = max(q3 - q1, self.eps)
            self.stats_[col] = {"median": med, "iqr": iqr}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, st in self.stats_.items():
            if col not in X.columns:
                continue
            s = pd.to_numeric(X[col], errors="coerce")
            X[col] = (s - st["median"]) / st["iqr"]
        return X


class ScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Scale numeric columns with different strategies.

    Strategies
    ----------
    - 'zscore'   : (x - mean) / std
    - 'minmax'   : (x - min) / (max - min)
    - 'range_0_10': 10 * (x - min) / (max - min)
    """
    def __init__(self, columns: Optional[List[str]] = None, strategy: str = "zscore", eps: float = 1e-12):
        self.columns = columns
        self.strategy = strategy
        self.eps = eps
        self.stats_: Dict[str, Dict[str, float]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.columns_ = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        self.stats_.clear()
        for col in self.columns_:
            s = X[col].astype(float)
            if self.strategy == "zscore":
                self.stats_[col] = {"mean": float(s.mean()), "std": float(s.std(ddof=0))}
            else:
                self.stats_[col] = {"min": float(s.min()), "max": float(s.max())}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "stats_")
        X = X.copy()
        for col in self.columns_:
            if col not in X.columns:
                continue
            s = X[col].astype(float)
            st = self.stats_[col]
            if self.strategy == "zscore":
                std = max(st["std"], self.eps)
                X[col] = (s - st["mean"]) / std
            elif self.strategy == "minmax":
                denom = max(st["max"] - st["min"], self.eps)
                X[col] = (s - st["min"]) / denom
            elif self.strategy == "range_0_10":
                denom = max(st["max"] - st["min"], self.eps)
                X[col] = 10.0 * (s - st["min"]) / denom
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        return X


# =============================================================================
# ===== 3) IMPUTATION =========================================================
# =============================================================================

class RandomNormalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing numeric values with samples from a normal distribution
    parameterized by TRAIN mean/std for each column.

    Parameters
    ----------
    columns : list[str] | None
        If None, all numeric columns are used.
    random_state : int | None
        Seed for reproducible sampling.

    Notes
    -----
    - Optional domain-specific lower bounds are enforced for selected columns
      (e.g., 'oldpeak' >= 0). Samples are redrawn until valid.
    - Simple rounding rules are applied to specific features (e.g., 'chol').
    """
    def __init__(self, columns: Optional[List[str]] = None, random_state: Optional[int] = None):
        self.columns = columns
        self.random_state = random_state
        self.stats_: Dict[str, tuple[float, float, float]] = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        self.stats_.clear()
        for col in self.columns:
            if X[col].isnull().any():
                mean = X[col].mean()
                std = X[col].std()

                # Domain-inspired lower bounds (extend as needed)
                if col == "chol":
                    lower_bound = 40
                elif col == "oldpeak":
                    lower_bound = 0
                else:
                    lower_bound = -np.inf

                self.stats_[col] = (mean, std, lower_bound)
        return self

    def transform(self, X):
        X = X.copy()
        rng = np.random.default_rng(self.random_state)

        for col, (mean, std, lower_bound) in self.stats_.items():
            n_missing = X[col].isna().sum()
            if n_missing > 0:
                # Generate values until all are valid
                valid_values: List[float] = []
                while len(valid_values) < n_missing:
                    sampled = rng.normal(loc=mean, scale=std, size=n_missing)
                    valid_sampled = sampled[sampled >= lower_bound]
                    valid_values.extend(valid_sampled.tolist())
                valid_values = valid_values[:n_missing]

                # Rounding rules by domain
                if col in ["trestbps", "chol", "thalch"]:
                    valid_values = [int(round(v)) for v in valid_values]
                elif col == "oldpeak":
                    valid_values = [round(v, 1) for v in valid_values]
                # Else: leave as continuous

                X.loc[X[col].isna(), col] = valid_values
        return X


# =============================================================================
# ===== 4) OUTLIER HANDLING ===================================================
# =============================================================================

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap extreme values using learned thresholds.

    Parameters
    ----------
    columns : list[str] | None
        If None, all numeric columns are used.
    method : {'iqr'}, default='iqr'
        Currently supports IQR method.
    factor : float, default=1.5
        Multiplier for IQR to define lower/upper caps.
    """
    def __init__(self, columns: Optional[List[str]] = None, method: str = "iqr", factor: float = 1.5):
        self.columns = columns
        self.method = method
        self.factor = factor
        self.caps_: Dict[str, tuple[float, float]] = {}

    def fit(self, X, y=None):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()

        for col in self.columns:
            if self.method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.factor * IQR
                upper = Q3 + self.factor * IQR
                self.caps_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.caps_.items():
            X[col] = X[col].clip(lower=lower, upper=upper)
        return X


# =============================================================================
# ===== 5) DISTRIBUTION SHAPING ===============================================
# =============================================================================

class SafeLog1p(BaseEstimator, TransformerMixin):
    """
    Apply log1p to selected columns (useful for heavy right skew).

    Notes
    -----
    - Preserves NaNs.
    - Clips to >= 0 before transform as a guard (negatives should be
      excluded upstream via cleaning/bounds).
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = np.log1p(X[col].clip(lower=0))
        return X
