from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scripts.eda.preprocessing.transformers import (
    RandomNormalImputer, OutlierCapper, SafeLog1p, ScalingTransformer
)

def build_preprocessing_pipeline() -> ImbPipeline:
    """
    Heart-failure dataset preprocessing pipeline (model-dependent).

        Steps:
            1) Impute numeric NaNs
            2) Cap outliers (IQR)
            3) Log1p skewed vars
            4) Min–max scale numeric (before SMOTE)
            5) SMOTE resampling (runs ONLY on fit_resample)
    """
    
    skew_cols = ["creatinine_phosphokinase", "serum_creatinine", "platelets"]
    numeric_cols = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
    numeric_to_scale = ["age", "ejection_fraction", "serum_sodium", "serum_creatinine", "time"]

    steps = [
        # fill NaN with synthetic random-normal values around the mean/std
        ("impute_numeric", RandomNormalImputer(random_state=42)),
        
        # cap extreme outliers based on IQR thresholds
        ("cap_outliers", OutlierCapper(columns=numeric_cols, method="iqr", factor=1.5)),

        # reduce skewness for heavily right-tailed variables
        ("log_skewed", SafeLog1p(columns=skew_cols)),

        # normalize features to [0, 1] range (important before SMOTE)
        ("scale_numeric", ScalingTransformer(columns=numeric_to_scale, strategy="minmax")),

        # combine SMOTEEN oversampling
        ("balance", SMOTEENN(
            sampling_strategy="auto",
            smote=SMOTE(k_neighbors=3, random_state=42),
            enn=EditedNearestNeighbours(n_neighbors=3)
        )),
    ]

    return ImbPipeline(steps=steps)
