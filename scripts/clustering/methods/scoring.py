from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure)


def internal_scores(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Compute internal clustering metrics.

    Assumes:
    - always >=2 clusters
    - sklearn metrics will not fail
    - no need to handle DBSCAN noise
    """

    labels = np.asarray(labels)
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "n_clusters": int(len(np.unique(labels))),
        "noise_%": float((labels == -1).mean() * 100.0),
    }


def external_scores(labels: np.ndarray, y_true: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """
    Compute external validation scores.

    Metrics:
    - ARI
    - NMI
    - homogeneity
    - completeness
    - v_measure
    """
    h, c, v = homogeneity_completeness_v_measure(y_true, labels)
    return {
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
    }


def score_internal_for_ranking(m: Dict[str, float]) -> tuple[float, float, float]:
    """
    Convert internal metrics into a tuple usable for sorting/ranking.

    Assumes:
    - silhouette, calinski_harabasz, davies_bouldin are always valid floats
    - no NaN / None values
    - no noise / 1-cluster edge cases
    """

    sil = float(m["silhouette"])
    ch  = float(m["calinski_harabasz"])
    db  = float(m["davies_bouldin"])
    s_db = 1.0 / (1.0 + db)     # DB lower is better → invert
    return sil, ch, s_db


def show_scores(
    name: str,
    X: np.ndarray,
    labels: np.ndarray,
    y_true: Optional[pd.Series] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Pretty-print internal and external metrics for a clustering result.

    Parameters
    ----------
    name : str
        Label for the experiment (e.g. 'PCA + KMeans').
    X : np.ndarray
        Feature matrix used for clustering.
    labels : np.ndarray
        Cluster assignments.
    y_true : Optional[pd.Series]
        Ground-truth labels, if available.
    """
    s_in = internal_scores(X, labels)
    s_ex = external_scores(labels, y_true)

    print(
        f"[{name}] -> "
        f"clusters={s_in['n_clusters']}, "
        f"noise={s_in['noise_%']:.1f}%, "
        f"sil={s_in['silhouette']}, "
        f"CH={s_in['calinski_harabasz']}, "
        f"DB={s_in['davies_bouldin']}, "
        f"ARI={s_ex.get('ARI')}, "
        f"NMI={s_ex.get('NMI')}, "
        f"V={s_ex.get('v_measure')}"
    )
    return {"internal": s_in, "external": s_ex}