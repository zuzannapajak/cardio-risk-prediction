"""
Dimensionality reduction helpers (e.g., K-selection on embeddings).
"""

from .selectors import SelectionResult, select_k_kmeans, select_k_agglomerative

__all__ = ["SelectionResult", "select_k_kmeans", "select_k_agglomerative"]
