import numpy as np
from sklearn.pipeline import Pipeline
from scripts.eda.cleaning.transformers import (ConvertToNumeric, InvalidValueToNaN, BoundsToNaN, EnsureBinaryInt)

def build_cleaning_pipeline() -> Pipeline:
    """
    Heart-failure dataset cleaning pipeline (model-agnostic).

        Steps:
            1) Convert selected columns to numeric (coerce errors)
            2) Replace invalid or out-of-range values with NaN
            3) Enforce binary columns to contain only {0, 1} integers

        Purpose:
            - Ensure correct datatypes and valid ranges before analysis.
            - This stage performs deterministic, rule-based cleaning
              and does NOT learn from the data (no leakage risk).
            - Suitable for full dataset cleaning and EDA.
    """

    # column groups
    numeric_cols = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
    binary_cols = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]
    all_cols = numeric_cols + binary_cols

    # physiologic plausibility bounds
    bounds = {
        "age": (18, 120),
        "ejection_fraction": (10, 85),
        "platelets": (20_000, 1_000_000),
        "serum_creatinine": (0.2, 15),
        "serum_sodium": (110, 160),
        "creatinine_phosphokinase": (0, 10_000),
        "time": (0, np.inf)
    }

    # cleaning steps
    steps = [
        # convert selected columns to numeric and coerce invalid values
        ("to_numeric", ConvertToNumeric(columns=all_cols)),

        # replace general invalid placeholders (e.g., negatives, non-finite) with NaN
        ("invalid_to_nan_generic", InvalidValueToNaN()),

        # replace values falling outside physiologic bounds with NaN
        ("invalid_to_nan_bounds", BoundsToNaN(bounds=bounds)),

        # enforce all binary columns to be valid {0, 1} integers
        ("enforce_binary", EnsureBinaryInt(columns=binary_cols)),
    ]

    return Pipeline(steps=steps)
