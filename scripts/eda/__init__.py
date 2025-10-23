"""
Exploratory Data Analysis Utilities
-----------------------------------

This package provides a structured set of tools to support exploratory data analysis (EDA),
statistical inspection, and visualization of datasets.

Modules
-------
plots.py
    Visualization utilities using matplotlib and seaborn for feature distributions,
    relationships, and class balance.

stats.py
    Numerical and statistical utilities for feature-level analysis, including
    skewness computation and mutual information with target variables.

reporting.py
    Lightweight reporting helpers that summarize dataset characteristics,
    unique values, class distributions, and create data dictionaries for export.
"""
from .plots import (
    create_plot_grid,
    plot_index_vs_columns,
    plot_violins,
    plot_violins_binary,
    plot_histograms_with_kde,
    plot_correlation_heatmap,
    plot_pairplot_with_hue,
    plot_class_distribution,
    plot_mutual_info,
)

from .stats import (
    calculate_skewness,
    compute_mutual_info,
)

from .reporting import (
    describe_unique_values,
    class_distribution_table,
    create_data_dictionary,
)

__all__ = [
    # Plotting
    "create_plot_grid",
    "plot_index_vs_columns",
    "plot_violins",
    "plot_histograms_with_kde",
    "plot_pairplot_with_hue",
    "plot_class_distribution",
    "plot_mutual_info",

    # Statistical analysis
    "calculate_skewness",
    "compute_mutual_info",

    # Reporting and summaries
    "describe_unique_values",
    "class_distribution_table",
    "create_data_dictionary",
]
