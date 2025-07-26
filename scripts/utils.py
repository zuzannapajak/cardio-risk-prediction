import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def create_plot_grid(n_plots, cols=3, figsize_per_plot=(5, 4)):
    """
    Creates a figure and grid of charts in a `cols` arrangement of columns and as many rows as needed.
    """
    rows = math.ceil(n_plots / cols)
    figsize = (figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    return fig, axes

def plot_index_column_relation(df):
    """
    Creates one figure with separate subplots to compare index with each numerical column.    
    """
    numerical_columns = df.select_dtypes(include=np.number)
    n_plots = len(numerical_columns)

    fig, axes = create_plot_grid(n_plots, cols=3)

    for i, col in enumerate(numerical_columns):
        ax = axes[i]
        ax.plot(df.index, df[col], marker='o', linestyle='-', alpha=0.7)
        ax.set_title(f"Index vs {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        ax.grid(True)

    # Hide unused plots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
