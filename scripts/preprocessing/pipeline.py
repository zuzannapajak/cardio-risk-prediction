from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.preprocessing.transformers import (
    OneHotEncoderTransformer,
    FixedMappingEncoderTransformer,
    ScalingTransformer,
    RobustScalingTransformer,
)

def build_preprocessing_pipeline() -> Pipeline:
    """
      - age: StandardScaler (z-score)
      - sex: Label map Female=0, Male=1 (else -1)
      - dataset: One-Hot
      - cp: One-Hot
      - trestbps: StandardScaler (z-score)
      - chol: StandardScaler (z-score)
      - fbs: Label map False=0, True=1, Missing=2
      - restecg: One-Hot (assumes literal "Missing" already present if missing)
      - thalch: StandardScaler (z-score)
      - exang: Label map False=0, True=1, Missing=2
      - oldpeak: RobustScaler (median/IQR)
      - slope: One-Hot (assumes literal "Missing" already present if missing)
      - num: target — untouched
    """
    steps = []

    # 1) Fix label mappings
    steps.append((
        "fixed_encoders",
        FixedMappingEncoderTransformer(
            mapping={
                "sex":   {"Female": 0, "Male": 1},
                "fbs":   {"False": 0, "True": 1, "Missing": 2},
                "exang": {"False": 0, "True": 1, "Missing": 2},
            },
            fallback_value=-1
        )
    ))

    # 2) One-Hot encoders
    steps.append((
        "onehot_dataset_cp",
        OneHotEncoderTransformer(columns=["dataset", "cp"], sparse=False)
    ))
    steps.append((
        "onehot_restecg_slope",
        OneHotEncoderTransformer(columns=["restecg", "slope"], sparse=False)
    ))

    # 3) Numeric scaling
    steps.append((
        "zscore_numeric",
        ScalingTransformer(columns=["age", "trestbps", "chol", "thalch"], strategy="zscore")
    ))
    steps.append((
        "robust_numeric",
        RobustScalingTransformer(columns=["oldpeak"])
    ))

    return Pipeline(steps=steps)