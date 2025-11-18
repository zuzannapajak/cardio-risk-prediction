"""
Dimensionality reduction utilities:
- PCA cumulative variance plot
- Interactive 3D embedding plots (Plotly)
- Embedding evaluation (trustworthiness + ROC-AUC probe)
"""

from .plots import plot_cumulative_variance, plot_3d
from .evaluation import eval_embedding

__all__ = [
    "plot_cumulative_variance",
    "plot_3d",
    "eval_embedding",
]
