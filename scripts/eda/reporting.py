import os
import pathlib
import numpy as np
import pandas as pd

def describe_unique_values(df):
    """
    Prints sorted unique values for numerical columns and unsorted unique values for categorical columns.
    Also prints the number of unique values for each column.
    
    Parameters:
    - df: pandas DataFrame
    """
    numerical_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(exclude=np.number).columns

    # Numerical columns
    for col in numerical_columns:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        print(f"\nColumn: {col}")
        print(unique_vals)
        print(f"{col}: {len(unique_vals)} unique values")

    # Categorical columns
    for col in categorical_columns:
        unique_vals = df[col].dropna().unique().tolist()
        print(f"\nColumn: {col}")
        print(unique_vals)
        print(f"{col}: {len(unique_vals)} unique values")


def class_distribution_table(
    df: pd.DataFrame,
    max_unique_values: int = 200
) -> pd.DataFrame:
    """
    Build a long-format table of class counts/percentages for categorical-like columns.
    
    Parameters:
    - df: pandas DataFrame  
        The input dataset containing categorical and/or numeric columns.
    - max_unique_values: int, default=200  
        The maximum number of unique values for a column to be considered "categorical-like."
        Columns with a smaller number of unique values (≤ this threshold) will be included
        even if their dtype is numeric.
    """
    out = []
    cat_cols = [
        c for c in df.columns
        if (df[c].dtype.name in ('object', 'category')) or (df[c].nunique(dropna=False) <= max_unique_values)
    ]

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False).sort_index()
        perc = df[col].value_counts(normalize=True, dropna=False).sort_index() * 100.0
        for cls in counts.index:
            out.append({
                "Column": col,
                "Class": str(cls),
                "Instances": int(counts.loc[cls]),
                "Percentage": float(round(perc.loc[cls], 4)),
            })

    return pd.DataFrame(out).sort_values(["Column", "Class"]).reset_index(drop=True)


def create_data_dictionary(
    df: pd.DataFrame,
    save_path: str | os.PathLike = "../outputs/data_dictionary/dictionary.csv"
) -> pd.DataFrame:
    """
    Create a compact data dictionary (dtype, % missing, uniques) and save to CSV.
    
    Parameters:
    - df: pandas DataFrame  
        The input dataset for which the data dictionary will be created.
    - save_path: str or os.PathLike, default="../outputs/data_dictionary/dictionary.csv"  
        Path where the resulting data dictionary CSV file will be saved.
        Intermediate directories will be created automatically if they do not exist.
    """
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "% Missing": (df.isnull().mean() * 100).round(4),
        "Unique Values": df.nunique(dropna=True)
    }).sort_values(by="Column").reset_index(drop=True)

    data_dict.to_csv(os.fspath(save_path), index=False)
    print(f"[INFO] Data dictionary saved to: {save_path}")
    return data_dict
