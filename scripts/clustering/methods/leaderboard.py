import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from typing import Dict, Any, List
from .gridsearch import (gridsearch_kmeans_params, gridsearch_agglomerative_params, gridsearch_dbscan)
from .scoring import internal_scores, score_internal_for_ranking


def build_global_leaderboard(
    datasets_map: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """
    Run grid-search for KMeans, Agglomerative and DBSCAN over multiple embeddings.

    Parameters
    ----------
    datasets_map : dict
        Mapping {embedding_name: X_embedded}.

    Returns
    -------
    list of dict
        Each entry: {
            "emb": <embedding_name>,
            "algo": "kmeans" | "agglomerative" | "dbscan",
            "params": {...},
            "labels": np.ndarray,
            "metrics": {...},   # internal_scores(...)
        }
    """
    board: List[Dict[str, Any]] = []

    for emb, X in datasets_map.items():
        # ----- KMEANS -----
        km_res = gridsearch_kmeans_params(X, ks=range(2, 6), n_inits=(10, 25))
        km_best = km_res.get("params") if isinstance(km_res, dict) else None
        if km_best is not None:
            n_clusters_km = km_best.get("n_clusters", km_best.get("k"))
            n_init = km_best.get("n_init", 10)
            km = KMeans(n_clusters=n_clusters_km, n_init=n_init)
            lab = km.fit_predict(X)
            m = internal_scores(X, lab)
        board.append({"emb": emb, "algo": "kmeans", "params": km_best, "labels": lab, "metrics": m})

        # ----- AGGLOMERATIVE -----
        ag_res  = gridsearch_agglomerative_params(X, ks=range(2, 6), linkages=("ward", "complete", "average"), metrics=("euclidean", "manhattan", "cosine"))
        ag_best = ag_res.get("params") if isinstance(ag_res, dict) else None
        if ag_best is not None:
            n_clusters_ag = ag_best.get("n_clusters", ag_best.get("k"))
            linkage = ag_best.get("linkage", "ward")
            metric = ag_best.get("metric", "euclidean")

            if linkage == "ward": ag = AgglomerativeClustering(n_clusters=n_clusters_ag,  linkage="ward")
            else: ag = AgglomerativeClustering(n_clusters=n_clusters_ag, linkage=linkage, metric=metric)
            lab = ag.fit_predict(X)
            m = internal_scores(X, lab)
        board.append({"emb": emb, "algo": "agglomerative", "params": ag_best, "labels": lab, "metrics": m})

        # ----- DBSCAN -----
        db_best = gridsearch_dbscan(X)
        if db_best and db_best.get("params"):
            board.append({"emb": emb, "algo": "dbscan", "params": db_best["params"], "labels": db_best["labels"], "metrics": db_best["metrics"]})

    return board


def rank_leaderboard(board: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort leaderboard entries by internal quality.

    Primary ordering: combined internal score (silhouette, CH, DB).
    Tie-breaker: prefer fewer clusters if available in metrics/params.
    """

    def key(row: Dict[str, Any]):
        m = row["metrics"]
        # prefer smaller k on ties
        k = (
            m.get("n_clusters")
            or (row["params"].get("k") if row.get("params") else None)
            or 0
        )
        return score_internal_for_ranking(m), -int(k)

    return sorted(board, key=key, reverse=True)
