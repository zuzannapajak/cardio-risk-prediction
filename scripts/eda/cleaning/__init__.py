"""
Cleaning utilities for the heart-failure dataset:
- Column type coercion
- Plausibility bounds → NaN
- Binary integrity checks
- Column dropping / rounding
- Categorical missing-value handling
"""

from .pipeline import build_cleaning_pipeline
from .transformers import (
    ConvertToNumeric,
    InvalidValueToNaN,
    BoundsToNaN,
    EnsureBinaryInt,
    DropColumns,
    DropColumnTransformer,
    RoundAndCastInt,
    DropHighNullColumns,
    CategoricalImputer,
    CategoricalMissingCategoryImputer,
)

__all__ = [
    "build_cleaning_pipeline",
    "ConvertToNumeric",
    "InvalidValueToNaN",
    "BoundsToNaN",
    "EnsureBinaryInt",
    "DropColumns",
    "DropColumnTransformer",
    "RoundAndCastInt",
    "DropHighNullColumns",
    "CategoricalImputer",
    "CategoricalMissingCategoryImputer",
]
