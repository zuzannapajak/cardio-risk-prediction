import pandas as pd
from scripts.transformers import (
    DropColumns,
    InvalidValueToNaN,
    DropHighNullColumns,
    OutlierCapper,
    RandomNormalImputer,
    CategoricalMissingCategoryImputer
)

def clean_dataframe(df: pd.DataFrame):
    
    # Drop id column
    df = DropColumns(columns=["id"]).fit_transform(df)
    
    # Drop high-null columns
    df = DropHighNullColumns(threshold=0.4).fit_transform(df)
    
    # Replace physiologically invalid values with NaN
    df = InvalidValueToNaN().fit_transform(df)
    
    # Impute missing values in numerical columns (random from normal distribution)
    df = RandomNormalImputer(random_state=42).fit_transform(df)
    
    # Impute missing values in categorical columns (adding new category "Missing")
    df = CategoricalMissingCategoryImputer().fit_transform(df)

    # Cap outliers using IQR method
    df = OutlierCapper().fit_transform(df)

    return df
