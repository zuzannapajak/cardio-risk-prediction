"""
Preprocessing utilities for model-ready features:
- Encoding (label/ordinal/one-hot/fixed mapping)
- Scaling & normalization
- Random-normal imputation
- Outlier capping
- Log1p transforms
"""

from .pipeline import build_preprocessing_pipeline
from .transformers import (
    LabelEncoderTransformer,
    OrdinalEncoderTransformer,
    OneHotEncoderTransformer,
    FixedMappingEncoderTransformer,
    UnitNormalizationTransformer,
    RobustScalingTransformer,
    ScalingTransformer,
    RandomNormalImputer,
    OutlierCapper,
    SafeLog1p,
)

__all__ = [
    "build_preprocessing_pipeline",
    "LabelEncoderTransformer",
    "OrdinalEncoderTransformer",
    "OneHotEncoderTransformer",
    "FixedMappingEncoderTransformer",
    "UnitNormalizationTransformer",
    "RobustScalingTransformer",
    "ScalingTransformer",
    "RandomNormalImputer",
    "OutlierCapper",
    "SafeLog1p",
]
