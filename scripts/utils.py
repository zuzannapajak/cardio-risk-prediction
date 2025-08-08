import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
import pathlib
import os
from sklearn.feature_selection import mutual_info_classif

def create_plot_grid(n_plots, cols=3, figsize_per_plot=(5, 4)):
    """
    Creates a matplotlib figure and grid of subplots arranged in 3 columns and as many rows as needed.

    Parameters:
        n_plots (int): Total number of subplots to create.
        cols (int): Number of columns in the grid (default is 3).
        figsize_per_plot (tuple): Size of each subplot (width, height).

    Returns:
        fig (Figure): The created matplotlib figure.
        axes (ndarray): Flattened array of subplot axes.
    """
    rows = math.ceil(n_plots / cols)
    figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    return fig, axes

def plot_index_column_relation(df):
    """
    Plots scatter subplots of index vs each numerical column in a single figure with 3 columns.

    Parameters:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        None
    """
    numerical_columns = df.select_dtypes(include='number').columns
    n_plots = len(numerical_columns)

    fig, axes = create_plot_grid(n_plots, cols=3)

    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        ax.scatter(df.index, df[col], s=40, alpha=0.6, marker='o', edgecolor='black', linewidth=0.5)
        ax.set_title(f"Index vs {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True)

    # Hide any unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def calculate_skewness(df, threshold=0.5, sort=True):
    """
    Computes skewness for all numerical columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Skewness threshold for flagging strong skew (default=0.5).
        sort (bool): Whether to sort the output by absolute skewness descending.

    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Skewness', 'Skew Type'].
    """
    skew_values = df.select_dtypes(include='number').skew()
    skew_df = pd.DataFrame({
        'Feature': skew_values.index,
        'Skewness': skew_values.values,
        'Skew Type': ['Right-skewed' if s > threshold else
                      'Left-skewed' if s < -threshold else
                      'Approximately symmetric'
                      for s in skew_values]
    })

    if sort:
        skew_df = skew_df.reindex(skew_df['Skewness'].abs().sort_values(ascending=False).index)

    return skew_df.reset_index(drop=True)

def plot_violins(df, cols=3, figsize_per_plot=(5, 4)):
    """
    Plots violin plots for all numerical columns in the DataFrame,
    arranged in a grid with `cols` columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (int): Number of columns in the grid.
        figsize_per_plot (tuple): Size of each subplot (width, height).
    """
    numeric_columns = df.select_dtypes(include='number').columns
    n_plots = len(numeric_columns)

    fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)

    for i, col in enumerate(numeric_columns):
        ax = axes[i]
        sns.violinplot(y=df[col], ax=ax, inner='box', linewidth=1, color='steelblue')
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_histograms_with_kde(df, cols=3, figsize_per_plot=(5, 4), bins=40):
    """
    Plots histograms with KDE (kernel density estimate) for all numerical columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (int): Number of columns in the grid layout.
        figsize_per_plot (tuple): Size of each subplot (width, height).
        bins (int): Number of histogram bins.
    """

    numerical_columns = df.select_dtypes(include=np.number).columns
    n_plots = len(numerical_columns)

    fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)

    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        sns.histplot(data=df, x=col, bins=bins, kde=True, ax=ax, color='blue', edgecolor='black')
        ax.set_title(f"{col}", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle("Histograms with KDE for Numerical Features", y=1.02, fontsize=14)
    plt.show()


def print_unique_values(df):
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


def plot_pairplot_with_hue(df, hue_col, title_prefix="Pairplot with distinction for"):
    """
    Generates a pairplot for selected numerical columns, colored by the given binary column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the dataset.
    - hue_col (str): Name of the binary column to distinguish in the plot (e.g., 'anaemia').
    - title_prefix (str): Optional prefix for the plot title.
    """
    df = df.copy()
    df[hue_col] = df[hue_col].astype('category')
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    sns.pairplot(df[numerical_columns + [hue_col]], hue=hue_col, corner=True)
    plt.suptitle(f"{title_prefix} {hue_col}", y=1.02)
    plt.show()
    
def compute_mutual_info(df, target_column):
    """
    Computes mutual information scores between numerical features and the target column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable (e.g., 'num').

    Returns:
        pd.DataFrame: Mutual information scores sorted descendingly.
    """

    df_numeric = df.select_dtypes(include=np.number)

    # drop rows with any NaNs in numerical features
    df_numeric = df_numeric.dropna()

    # separate features and target
    X = df_numeric.drop(columns=[target_column])
    y = df_numeric[target_column]

    # compute MI
    mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=0)

    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    }).sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    return mi_df

def plot_mutual_info(mi_df, figsize=(10, 6)):
    """
    Plots a barplot of mutual information scores.

    Parameters:
        mi_df (pd.DataFrame): Output of `compute_mutual_info`.
        figsize (tuple): Size of the figure.
    """
    plt.figure(figsize=figsize)
    sns.barplot(data=mi_df, x='Mutual Information', y='Feature', color=sns.color_palette('viridis')[3])
    plt.title('Mutual Information Scores')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_class_distribution(df, cols=3, figsize_per_plot=(5, 4), max_unique_values=20):
    """
    Plots multiple class distributions in a grid layout.

    Parameters:
    - df: pandas DataFrame containing the data.
    - cols: number of columns in the grid.
    - figsize_per_plot: size of each subplot
    - max_unique_values: maximum number of unique values to consider a column as categorical.
    """

    # Automatically select categorical columns (object, category, or with few unique values)
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype in ['object', 'category'] or df[col].nunique() <= max_unique_values
    ]

    if not categorical_cols:
        print("No categorical columns found to plot.")
        return

    n_plots = len(categorical_cols)
    fig, axes = create_plot_grid(n_plots, cols=cols, figsize_per_plot=figsize_per_plot)

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        
        class_counts = df[col].value_counts()
        class_counts.index = class_counts.index.astype(str)
        class_counts = class_counts.sort_index()
        
        class_percentages = df[col].value_counts(normalize=True) * 100
        class_percentages.index = class_percentages.index.astype(str)
        class_percentages = class_percentages.sort_index()
        
        summary_df = pd.DataFrame({
            'Class': class_counts.index.astype(str),
            'Instances': class_counts.values,
            'Percentage': class_percentages.values
        })

        print(f"\n{'='*60}\nClass Distribution Summary for '{col}':")
        print(summary_df.to_markdown(index=False, floatfmt=".2f"))

        sns.barplot(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            ax=ax,
            hue=class_counts.index.astype(str),
            palette='viridis',
            legend=False
        )
        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel('Class')
        ax.set_ylabel('Instances')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def create_data_dictionary(df: pd.DataFrame, save_path: str = "../outputs/data_dictionary/dictionary.csv") -> pd.DataFrame:
    """
    Creates a compact table of dataset columns with dtype, % missing, and unique counts,
    and saves it to CSV.
    """
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "% Missing": (df.isnull().mean() * 100).round(4),
        "Unique Values": df.nunique()
    }).sort_values(by="Column").reset_index(drop=True)

    data_dict.to_csv(os.fspath(save_path), index=False)
    print(f"Data dictionary saved to: {save_path}")
    return data_dict