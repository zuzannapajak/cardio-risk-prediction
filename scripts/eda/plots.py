from typing import List, Optional, Tuple, Union
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

def plot_violins_binary(
    df: pd.DataFrame,
    binary_cols: List[str],
    cols: int = 6,
    figsize_per_plot: Tuple[float, float] = (5, 4),
    numeric_cols: Optional[List[str]] = None,
    dropna: bool = True,
) -> None:
    """
    For each numeric column (target on y), plot violins grouped by each binary column (x).

    Parameters
    ----------
    df : pd.DataFrame
    binary_cols : list[str]
        Binary columns used on the x-axis (e.g., 0/1).
    cols : int
        Number of columns in each subplot grid.
    figsize_per_plot : (float, float)
        Size of each subplot.
    numeric_cols : list[str] | None
        Numeric columns to use as targets. If None, auto-detect numeric cols excluding `binary_cols`.
    dropna : bool
        Drop rows with NA in (target, feature) before plotting.
    """
    # choose numeric targets
    if numeric_cols is None:
        numeric_cols = [
            c for c in df.select_dtypes(include='number').columns
            if c not in set(binary_cols)
        ]
    if not numeric_cols or not binary_cols:
        return

    # one figure per numeric target, grid over all binaries
    for target in numeric_cols:
        n_plots = len(binary_cols)
        fig, axes = create_plot_grid(n_plots, cols, figsize_per_plot)

        for i, feat in enumerate(binary_cols):
            ax = axes[i]
            sub = df[[target, feat]]
            if dropna:
                sub = sub.dropna()

            # Ensure x shows 0 → 1 when possible
            x_order = [0, 1]
            uniq = sorted(pd.unique(sub[feat]))
            if set(uniq) != set(x_order):
                x_order = uniq

            sub = sub.copy()
            sub[feat] = pd.Categorical(sub[feat], categories=x_order, ordered=True)
            
            sns.violinplot(data=sub, x=feat, y=target, inner='box', linewidth=1, ax=ax)
            ax.set_title(f"{target} | {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel(target)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        # hide any unused axes
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

def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: None,
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    title: str = 'Correlation Matrix Heatmap',
) -> None:
    """
    Plot a correlation matrix heatmap for selected (or all numeric) columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    cols can be:
      - list/Index/ndarray of column names,
      - boolean mask (Series/ndarray) aligned to df.columns,
      - DataFrame (uses its columns).
    figsize : (int, int)
        Figure size in inches (width, height).
    annot : bool
        Whether to show correlation values inside the heatmap.
    title : str
        Title of the heatmap.
    """
    if cols is None:
        data = df.select_dtypes(include='number')
    else:
           if isinstance(cols, pd.DataFrame):
            cols = cols.columns

            # Boolean mask case
            if isinstance(cols, (pd.Series, np.ndarray)) and getattr(cols, 'dtype', None) == bool:
                data = df.loc[:, cols]
            else:
                # Treat as column-name iterable
                cols = list(cols)
                # intersect with df.columns to avoid KeyError
                cols = [c for c in cols if c in df.columns]
                if not cols:
                    raise ValueError("No valid columns found in `cols`.")
                data = df.loc[:, cols]

            # Keep only numeric
            data = data.select_dtypes(include='number')

    if data.shape[1] == 0:
        raise ValueError("No numeric columns available for correlation after filtering.")

    corr = data.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, linewidths=0.5, fmt='.2f', cmap='coolwarm')
    plt.title(title)
    plt.show()

def plot_pairplot_with_hue(
    df: pd.DataFrame,
    hue_cols: list[str] | str,
    title_prefix: str = "Pairplot with distinction for",
    sample: int | None = None,
) -> None:
    """
    Generate Seaborn pairplots for numeric columns colored by one or multiple hue columns.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    hue_cols : list[str] | str
        Column name or list of binary columns to distinguish in the plots (e.g., ['anaemia', 'diabetes']).
    title_prefix : str
        Optional prefix for the plot title.
    sample : int | None
        Optional subsample size for faster plotting.
    """

    data = df.copy()
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # Count unique non-null values per numeric column
    nunq = {c: data[c].dropna().nunique() for c in numeric_cols}
    nonbinary_vars = [c for c in numeric_cols if nunq[c] > 2]

    for hue_col in hue_cols:
        if hue_col not in data.columns:
            print(f"Skipping '{hue_col}' — not in DataFrame.")
            continue

        vars_for_plot = [c for c in nonbinary_vars if c != hue_col]
        sub = data[vars_for_plot + [hue_col]].copy()
        sub[hue_col] = sub[hue_col].astype("category")

        g = sns.pairplot(sub, vars=vars_for_plot, hue=hue_col, corner=True, diag_kind="kde", dropna=True)
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
