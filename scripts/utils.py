import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns

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
    Plots subplots of index vs each numerical column in a single figure with 3 columns.

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
        ax.plot(df.index, df[col], marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Index vs {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True)

    # Hide any unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
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
    
def plot_class_distribution(df, target_columns, cols=3, figsize_per_plot=(5, 4)):
    """
    Plots multiple class distributions in a grid layout.

    Parameters:
    - df: pandas DataFrame containing the data.
    - target_columns: list of column names to plot.
    - cols: number of columns in the grid.
    - figsize_per_plot: size of each subplot.
    """

    n_plots = len(target_columns)
    fig, axes = create_plot_grid(n_plots, cols=cols, figsize_per_plot=figsize_per_plot)

    for i, col in enumerate(target_columns):
        ax = axes[i]
        class_counts = df[col].value_counts().sort_index()
        class_percentages = df[col].value_counts(normalize=True).sort_index() * 100

        summary_df = pd.DataFrame({
            'Class': class_counts.index.astype(str),
            'Instances': class_counts.values,
            'Percentage': class_percentages.values
        })

        print(f"\n{'='*60}\nClass Distribution Summary for '{col}':")
        print(summary_df.to_markdown(index=False, floatfmt=".2f"))

        sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, ax=ax, hue=class_counts.index.astype(str), palette='viridis', legend=False)
        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel('Class')
        ax.set_ylabel('Instances')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()