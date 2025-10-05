from typing import Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Layout

def create_plot_grid(n_plots: int, cols: int = 3, figsize_per_plot: Tuple[float, float] = (5, 4)):
    """
    Create a grid of subplots sized by number of plots and columns.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Flattened array of axes with length >= n_plots. Unused axes are returned too.
    """
    rows = max(1, math.ceil(n_plots / cols))
    figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.array([axes])
    return fig, axes


# Plots

def plot_index_vs_columns(df: pd.DataFrame, cols: int = 3, figsize_per_plot: Tuple[float, float] = (5, 4)) -> None:
    """
    Scatter plots: index vs each numeric column.
    
    Parameters:
        df (DataFrame): Input pandas DataFrame.

    Returns:
        None
    """
    numerical_columns = df.select_dtypes(include='number').columns.tolist()
    n_plots = len(numerical_columns)
    if n_plots == 0:
        return

    fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)
    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        ax.scatter(df.index, df[col], s=20, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_title(f"Index vs {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_violins(df: pd.DataFrame, cols: int = 3, figsize_per_plot: Tuple[float, float] = (5, 4)) -> None:
    """
    Violin plots for all numeric columns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (int): Number of columns in the grid.
        figsize_per_plot (tuple): Size of each subplot (width, height).
    """
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    n_plots = len(numeric_columns)
    if n_plots == 0:
        return

    fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)
    for i, col in enumerate(numeric_columns):
        ax = axes[i]
        sns.violinplot(y=df[col], ax=ax, inner='box', linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_histograms_with_kde(
    df: pd.DataFrame,
    cols: int = 3,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    bins: int = 40
) -> None:
    """
    Histograms + KDE for numeric columns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        cols (int): Number of columns in the grid layout.
        figsize_per_plot (tuple): Size of each subplot (width, height).
        bins (int): Number of histogram bins.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_plots = len(num_cols)
    if n_plots == 0:
        return

    fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)
    for i, col in enumerate(num_cols):
        ax = axes[i]
        sns.histplot(data=df, x=col, bins=bins, kde=True, ax=ax, color='blue', edgecolor='black')
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.suptitle("Histograms with KDE for Numerical Features", y=1.02, fontsize=14)
    plt.show()


def plot_pairplot_with_hue(
    df: pd.DataFrame,
    hue_col: str,
    title_prefix: str = "Pairplot with distinction for",
    sample: int | None = None,
) -> None:
    """
    Pairplot for numeric columns colored by `hue_col`. Optionally subsample for speed.

    Parameters:
        - df (pd.DataFrame): DataFrame containing the dataset.
        - hue_col (str): Name of the binary column to distinguish in the plot (e.g., 'anaemia').
        - title_prefix (str): Optional prefix for the plot title.
    """
    data = df.copy()
    if sample is not None and len(data) > sample:
        data = data.sample(n=sample, random_state=0)

    if hue_col in data.columns:
        data[hue_col] = data[hue_col].astype('category')

    num_cols = data.select_dtypes(include=np.number).columns.tolist()
    g = sns.pairplot(data[num_cols + [hue_col]] if hue_col in data else data[num_cols],
                     hue=hue_col if hue_col in data else None,
                     corner=True)
    plt.suptitle(f"{title_prefix} {hue_col}", y=1.02)
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


def plot_mutual_info(mi_df: pd.DataFrame, figsize: Tuple[float, float] = (10, 6)) -> None:
    """
    Horizontal barplot for mutual information scores (expects columns: Feature, Mutual Information).
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable (e.g., 'num').

    Returns:
        pd.DataFrame: Mutual information scores sorted descendingly.
    """
    if mi_df.empty:
        return
    plt.figure(figsize=figsize)
    sns.barplot(data=mi_df, x='Mutual Information', y='Feature', color=sns.color_palette('viridis')[3])
    plt.title('Mutual Information Scores')
    plt.xlabel('Score')
    plt.ylabel('Feature')
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
