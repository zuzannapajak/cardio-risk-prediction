from typing import List
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.preprocessing.transformers import (
    FixedMappingEncoderTransformer,
    LabelEncoderTransformer,
    ScalingTransformer,
)

def build_dimred_preprocessing_pipeline() -> Pipeline:
    """
    Preprocessing tailored for dimensionality reduction:
      - sex: Label map Female=0, Male=1 (else -1)
      - fbs: Label map False=0, True=1, Missing=2
      - exang: Label map False=0, True=1, Missing=2
      - dataset, cp, restecg, slope: Label-encoded (stable order)
      - age, trestbps, chol, thalch, oldpeak: StandardScaler (z-score)
      - num: target — untouched
    """
    steps = []

    # 1) Fixed mappings for simple binary flags
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

    # 2) Label-encode nominal columns (categorical → integer codes)
    steps.append((
        "label_encoders",
        LabelEncoderTransformer(
            columns=["dataset", "cp", "restecg", "slope"]
        )
    ))

    # 3) Scale numeric columns (z-score)
    steps.append((
        "zscore_scaling",
        ScalingTransformer(
            columns=["age", "trestbps", "chol", "thalch", "oldpeak"],
            strategy="zscore"
        )
    ))

    # Note: 'num' (target variable) remains unchanged and is excluded.

    return Pipeline(steps=steps)
