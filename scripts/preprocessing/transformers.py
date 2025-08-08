from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder as _SkOrdinalEncoder, OneHotEncoder
from inspect import signature
from sklearn.utils.validation import check_is_fitted

  
#==== Label encoders====
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Label-encode specific categorical columns with stable mappings.
    Unseen categories at transform time -> -1.
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
    Ordinal-encode columns given an explicit order for each.
    Unknown categories -> -1.
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
    def __init__(self, columns, sparse=False, drop_original=True):
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

        # Ensure numeric dummies
        oh_df = oh_df.astype(np.float64)
        return pd.concat([X, oh_df], axis=1)

class FixedMappingEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Encode categorical columns using fixed, explicit mappings.
    Unmapped / unseen values -> fallback_value (default: -1).
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

#==== Normalization ====
class UnitNormalizationTransformer(BaseEstimator, TransformerMixin):
    """
    Apply unit conversions / simple scaling factors per column.
    Example: {'age_days': 1/365} -> converts days to years.
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
    
#==== Scaling transformers ====
class ScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Scale numeric columns with different strategies:
      - 'zscore': (x - mean) / std
      - 'minmax': (x - min) / (max - min)
      - 'range_0_10': min-max to [0, 10]
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
    
class RobustScalingTransformer(BaseEstimator, TransformerMixin):
    """
    Robust scaling per column: (x - median) / IQR
    where IQR = Q3 - Q1. Falls back to no-op if IQR ~ 0.
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