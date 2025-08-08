import pandas as pd
from sklearn.pipeline import Pipeline
from scripts.cleaning.transformers import (
    DropColumns,
    InvalidValueToNaN,
    DropHighNullColumns,
    OutlierCapper,
    RandomNormalImputer,
    CategoricalMissingCategoryImputer
)

def build_cleaning_pipeline(null_threshold: float = 0.40) -> Pipeline:
    """
    1) Drop id column
    2) Drop high-null columns
    3) Invalid -> NaN (rule-based)
    4) Impute numerics (random normal around mean/std)
    5) Impute categoricals (add 'Missing')
    6) Cap numeric outliers with IQR method
    """
    steps = []

    # 1) Drop id column
    steps.append(("drop_id", DropColumns(columns=["id"])))

    # 2) Drop high-null columns
    steps.append(("drop_high_nulls", DropHighNullColumns(threshold= null_threshold)))

    # 3) Convert invalid values to NaN
    steps.append(("invalid_to_nan", InvalidValueToNaN()))

    # 4) Impute numeric columns
    steps.append(("impute_numeric", RandomNormalImputer(random_state=42)))

    # 5) Impute categorical columns with 'Missing'
    steps.append(("impute_categorical", CategoricalMissingCategoryImputer(fill_value="Missing")))

    # 6) Cap numeric outliers via IQR
    steps.append(("cap_outliers", OutlierCapper()))

    return Pipeline(steps=steps)