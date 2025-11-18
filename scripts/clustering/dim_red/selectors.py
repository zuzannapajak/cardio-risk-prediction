import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering
from dataclasses import dataclass
from typing import Dict, Optional, Iterable
from ..methods.scoring import internal_scores
from ..plotting.plot_selection import plot_k_selection_curves


@dataclass
class SelectionResult:
    """
    Result of K-selection for a given algorithm and embedding.

    Attributes
    ----------
    df_scores : DataFrame
        Metrics per k.
    best_by_metric : dict
        Mapping metric_name -> best k.
    extra : dict or None
        Additional information (e.g., linkage, metric).
    """
    df_scores: pd.DataFrame
    best_by_metric: Dict[str, int]
    extra: Optional[Dict] = None


def select_k_kmeans(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 21),
    n_init: int = 10,
    random_state: int = 42,
    data_name: Optional[str] = None,
) -> SelectionResult:
    """
    Perform K-selection for KMeans using internal metrics and inertia.

    Parameters
    ----------
    X : array-like
        Data or embedding to cluster.
    k_range : iterable of int
        Candidate values of k.
    n_init : int
        Number of KMeans initializations.
    random_state : int
        Random seed.
    data_name : str, optional
        Name of the dataset/embedding, used in the plot.

    Returns
    -------
    SelectionResult
        Contains df_scores and best k per metric.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        met = internal_scores(X, labels)
        rows.append({"k": k, "inertia": float(km.inertia_), **met})

    df = pd.DataFrame(rows)
    best = {
        "inertia": int(df.loc[df["inertia"].idxmin(), "k"]),
        "silhouette": int(df.loc[df["silhouette"].idxmax(), "k"]),
        "calinski_harabasz": int(df.loc[df["calinski_harabasz"].idxmax(), "k"]),
        "davies_bouldin": int(df.loc[df["davies_bouldin"].idxmin(), "k"]),
    }

    tag = data_name.upper() if data_name else None
    plot_k_selection_curves(df, title="KMeans model selection", include_inertia=True, tag=tag)
    return SelectionResult(df_scores=df, best_by_metric=best)

def select_k_agglomerative(
    X: np.ndarray,
    k_range: Iterable[int] = range(2, 21),
    linkage: str = "ward",
    metric: Optional[str] = None,
    data_name: Optional[str] = None,
) -> SelectionResult:
    """
    Perform K-selection for AgglomerativeClustering using internal metrics.

    Parameters
    ----------
    X : array-like
        Data or embedding to cluster.
    k_range : iterable of int
        Candidate values of k.
    linkage : {'ward', 'complete', 'average', ...}
        Linkage strategy.
    metric : str, optional
        Distance metric; for 'ward' scikit-learn uses 'euclidean'.
    data_name : str, optional
        Name of the dataset/embedding, used in the plot.

    Returns
    -------
    SelectionResult
        Contains df_scores and best k per metric, plus extra info.
    """
    rows = []
    for k in k_range:
        if linkage == "ward": agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        else: agg = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric or "euclidean")
        labels = agg.fit_predict(X)
        met = internal_scores(X, labels)
        rows.append({"k": k, **met})

    df = pd.DataFrame(rows)
    best = {
        "silhouette": int(df.loc[df["silhouette"].idxmax(), "k"]),
        "calinski_harabasz": int(df.loc[df["calinski_harabasz"].idxmax(), "k"]),
        "davies_bouldin": int(df.loc[df["davies_bouldin"].idxmin(), "k"]),
    }

    eff_metric = metric or ("euclidean" if linkage != "ward" else "euclidean")
    tag = data_name.upper() if data_name else None
    plot_k_selection_curves(df, title=f"Agglomerative model selection (linkage={linkage}, metric={eff_metric})", include_inertia=False, tag=tag)
    extra = {"linkage": linkage, "metric": eff_metric}
    return SelectionResult(df_scores=df, best_by_metric=best, extra=extra)
