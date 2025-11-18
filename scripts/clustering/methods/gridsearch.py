from typing import Dict, Optional, Iterable, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from .scoring import internal_scores


def _pick_best(df_metrics: pd.DataFrame) -> Optional[int]:
    """
    Rank rows by internal metrics and return index of the best configuration.

    Ranking strategy:
    - Each metric is ranked (silhouette ↓, CH ↓, DB ↑)
    - Sum of ranks is minimized
    """
    g = df_metrics.copy()
    g["r_sil"] = g["silhouette"].rank(ascending=False, method="min")
    g["r_ch"] = g["calinski_harabasz"].rank(ascending=False, method="min")
    g["r_db"] = g["davies_bouldin"].rank(ascending=True, method="min")
    g["rank_sum"] = g[["r_sil", "r_ch", "r_db"]].sum(axis=1)

    g = g.dropna(subset=["rank_sum"])
    return int(g.sort_values("rank_sum").index[0])


def gridsearch_kmeans_params(
    X: np.ndarray,
    ks: Iterable[int] = range(2, 11),
    n_inits: Iterable[int] = (10, 25),
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Grid-search over (k, n_init) for KMeans using internal metrics.

    Returns
    -------
    dict
        {"params": {"n_clusters": k_best, "n_init": n_init_best}}.
    """
    rows = []
    for k in ks:
        for n_init in n_inits:
            km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            labels = km.fit_predict(X)
            m = internal_scores(X, labels)
            rows.append({"k": k, "n_init": n_init, **m})

    df = pd.DataFrame(rows)
    idx = _pick_best(df)
    best = df.loc[idx]
    return {"params": {"n_clusters": int(best["k"]), "n_init": int(best["n_init"])}}


def gridsearch_agglomerative_params(
    X: np.ndarray,
    ks: Iterable[int] = range(2, 11),
    linkages: Iterable[str] = ("ward", "complete", "average"),
    metrics: Iterable[str] = ("euclidean", "manhattan", "cosine"),
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Grid-search AgglomerativeClustering over (k, linkage, metric).

    - For linkage='ward', only 'euclidean' is allowed by scikit-learn.

    Returns
    -------
    dict
        {"params": {"n_clusters": ..., "linkage": ..., "metric": ...}}.
    """
    rows = []
    for k in ks:
        for linkage in linkages:
            metrics_to_try = ("euclidean",) if linkage == "ward" else tuple(metrics)
            for metric in metrics_to_try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
                labels = agg.fit_predict(X)
                m = internal_scores(X, labels)
                rows.append({"k": k, "linkage": linkage, "metric": metric, **m})

    df = pd.DataFrame(rows)
    idx = _pick_best(df)
    best = df.loc[idx]
    return {"params": {"n_clusters": int(best["k"]), "linkage": best["linkage"], "metric": best["metric"]}}

def gridsearch_dbscan(
    X: np.ndarray,
    eps_grid: Iterable[float] = (0.2, 0.3, 0.5, 0.8, 1.2),
    min_samples_grid: Iterable[int] = (3, 5, 10),
) -> Optional[Dict[str, Any]]:
    """
    Simple grid-search for DBSCAN over eps and min_samples, using silhouette score as the criterion.
    Returns
    -------
    dict
        {
            "params": {"eps": ..., "min_samples": ...},
            "metrics": {...},
            "labels": np.ndarray
        }
    """
    best = None
    best_score = -np.inf

    for eps in eps_grid:
        for ms in min_samples_grid:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            if len(np.unique(labels)) < 2: continue

            m = internal_scores(X, labels)
            score = m.get("silhouette", np.nan)
            cand = {"params": {"eps": eps, "min_samples": ms}, "metrics": m, "labels": labels}

            if score > best_score:
                best = cand
                best_score = score

    return best