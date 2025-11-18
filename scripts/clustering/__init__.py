"""
High-level utilities for clustering analysis and visualization.

This package provides:
- Clustering model selection and grid-search utilities
- Internal / external clustering metrics
- Global leaderboard building and ranking
- Cluster profiling (SMDs, top-N features)
- 3D visualization and SMD plots
- Dimensionality reduction K-selection helpers
"""

from .methods.gridsearch import (gridsearch_kmeans_params, gridsearch_agglomerative_params, gridsearch_dbscan)
from .methods.scoring import internal_scores, external_scores, score_internal_for_ranking, show_scores
from .methods.leaderboard import (build_global_leaderboard, rank_leaderboard)
from .methods.profiling import (compute_smds, top_union_smd_table)

from .plotting.plot_clusters import plot_clusters_target_3d
from .plotting.plot_smd import plot_smd_stacked_bars, plot_smd_heatmap

from .dim_red.selectors import (SelectionResult, select_k_kmeans, select_k_agglomerative)

__all__ = [
    # grid-search
    "gridsearch_kmeans_params",
    "gridsearch_agglomerative_params",
    "gridsearch_dbscan",
    # scoring
    "internal_scores",
    "external_scores",
    "score_internal_for_ranking",
    "show_scores",
    # leaderboard
    "build_global_leaderboard",
    "rank_leaderboard",
    # profiling
    "compute_smds",
    "top_union_smd_table",
    # plotting
    "plot_clusters_target_3d",
    "plot_smd_stacked_bars",
    "plot_smd_heatmap",
    # dimred
    "SelectionResult",
    "select_k_kmeans",
    "select_k_agglomerative",
]
