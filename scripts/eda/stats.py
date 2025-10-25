import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def calculate_skewness(
    df: pd.DataFrame,
    threshold: float = 0.5,
    sort: bool = True
) -> pd.DataFrame:
    """
    Compute skewness of all numeric columns and classify type.
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Skewness threshold for flagging strong skew (default=0.5).
        sort (bool): Whether to sort the output by absolute skewness descending.

    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Skewness', 'Skew Type'].
    """
    skew_values = df.select_dtypes(include='number').skew(numeric_only=True)
    skew_df = pd.DataFrame({
        'Feature': skew_values.index,
        'Skewness': skew_values.values,
        'Skew Type': [
            'Right-skewed' if s > threshold else
            'Left-skewed' if s < -threshold else
            'Approximately symmetric'
            for s in skew_values
        ],
    })
    if sort:
        skew_df = skew_df.reindex(skew_df['Skewness'].abs().sort_values(ascending=False).index)
    return skew_df.reset_index(drop=True)


def compute_mutual_info(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Mutual information between numeric features and a (numeric) target.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable (e.g., 'num').

    Returns:
        pd.DataFrame: Mutual information scores sorted descendingly.
    
    Notes
    -----
    - Drops rows with NaNs in numeric features/target.
    - If your target is categorical (encoded), ensure it's in numeric form.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target '{target_column}' not found in DataFrame.")

    df_num = df.select_dtypes(include=np.number)
    if target_column not in df_num.columns:
        raise ValueError("Target must be numeric for mutual_info_classif in this helper.")

    # Drop rows with any NaNs across numeric features/target
    df_num = df_num.dropna(axis=0, how='any')

    X = df_num.drop(columns=[target_column])
    y = df_num[target_column].astype(int) if df_num[target_column].dtype.kind not in 'iu' else df_num[target_column]

    if X.shape[1] == 0:
        return pd.DataFrame(columns=['Feature', 'Mutual Information'])

    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=random_state)
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores}) \
            .sort_values(by='Mutual Information', ascending=False) \
            .reset_index(drop=True)
    return mi_df
