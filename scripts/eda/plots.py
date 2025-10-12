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

def plot_class_distribution(
    data,
    columns=None,                # None => all columns; str => one column; list[str] => selected
    grid_cols=3,                 # number of subplot columns in the grid
    figsize_per_plot=(5, 4),
    max_unique_values=20,        # threshold to treat as categorical
    bins=30                      # bins for numeric histograms
):
    """
    Plots distributions for selected columns (or all if not provided).
    - Categorical-like columns (dtype object/category or <= max_unique_values unique): bar plot of counts.
    - Numeric-like columns: histogram.
    
    Parameters
    ----------
    data : pd.DataFrame | pd.Series | array-like
    columns : None | str | list[str]
        Columns to plot. If None, all columns are plotted.
    grid_cols : int
        Number of columns in the subplot grid.
    figsize_per_plot : (w, h)
        Size of each subplot.
    max_unique_values : int
        Max unique values to consider a column categorical.
    bins : int
        Number of bins for numeric histograms.
    """

    # input to DataFrame
    if isinstance(data, pd.Series):
        df = pd.DataFrame({data.name if data.name else 'value': data})
    elif isinstance(data, (list, tuple, np.ndarray)):
        df = pd.DataFrame({'value': data})
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    # columns to plot
    if columns is None:
        cols_to_plot = list(df.columns)
    elif isinstance(columns, str):
        if columns not in df.columns:
            raise KeyError(f"Column '{columns}' not found in data.")
        cols_to_plot = [columns]
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found: {missing}")
        cols_to_plot = list(columns)

    if len(cols_to_plot) == 0:
        print("No columns to plot.")
        return

    # grid
    n_plots = len(cols_to_plot)
    grid_rows = math.ceil(n_plots / grid_cols)
    fig_width = figsize_per_plot[0] * grid_cols
    fig_height = figsize_per_plot[1] * grid_rows
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # plot each column
    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        s = df[col].dropna()

        is_categorical = (
            s.dtype.name in ['object', 'category', 'bool']
            or s.nunique(dropna=True) <= max_unique_values
        )

        if is_categorical:
            counts = s.astype(str).value_counts().sort_index()
            perc = s.astype(str).value_counts(normalize=True).sort_index() * 100

            summary_df = pd.DataFrame({
                'Class': counts.index,
                'Instances': counts.values,
                'Percentage': np.round(perc.values, 2)
            })
            print(f"\n{'='*60}\nDistribution summary for '{col}':")
            print(summary_df.to_markdown(index=False, floatfmt=".2f"))

            sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, palette='viridis', legend=False)
            ax.set_ylabel('Instances')
            ax.set_xlabel('Class')

        else:
            desc = s.describe()[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            print(f"\n{'='*60}\nNumeric summary for '{col}':")
            print(desc.to_frame(name=col).to_markdown(floatfmt=".3f"))

            sns.histplot(s, bins=bins, ax=ax)
            ax.set_ylabel('Frequency')
            ax.set_xlabel(col)

        ax.set_title(str(col), fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

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
