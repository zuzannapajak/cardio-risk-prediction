import umap, numpy as np
import pandas as pd, seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Iterable, Dict, Optional
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, homogeneity_completeness_v_measure, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scripts.dim_red.plots import plot_3d
import plotly.graph_objects as go
from dataclasses import dataclass
import matplotlib.pyplot as plt

def build_global_leaderboard(datasets_map):
    """
    Collect (embedding × algo) candidates with metrics using your existing utilities.

    Works with:
        - gridsearch_kmeans_params(X) -> {"params": {..., "n_clusters": ...}} or {..., "k": ...}
        - gridsearch_agglomerative_params(X) -> {"params": {..., "n_clusters": ...}} or {..., "k": ...}
        - gridsearch_dbscan(X) -> {"params": ..., "labels": ..., "metrics": ...}
    """
    board = []

    for emb, X in datasets_map.items():
        # ----- KMEANS -----
        km_res = gridsearch_kmeans_params(X, ks=range(2, 6), n_inits=(10, 25))
        km_best = km_res.get("params") if isinstance(km_res, dict) else None
        if km_best:
            # support both new ("n_clusters") and old ("k") style
            n_clusters_km = km_best.get("n_clusters", km_best.get("k"))
            km = build_model("kmeans", {"params": km_best, "n_clusters": n_clusters_km})
            lab = km.fit_predict(X)
            m = internal_scores(X, lab)
            board.append({
                "emb": emb,
                "algo": "kmeans",
                "params": km_best,
                "labels": lab,
                "metrics": m,
            })

        # ----- AGGLOMERATIVE -----
        ag_res = gridsearch_agglomerative_params(
            X,
            ks=range(2, 6),
            linkages=("ward", "complete", "average"),
            metrics=("euclidean", "manhattan", "cosine"),
        )
        ag_best = ag_res.get("params") if isinstance(ag_res, dict) else None
        if ag_best:
            # support both "n_clusters" and "k"
            n_clusters_ag = ag_best.get("n_clusters", ag_best.get("k"))
            ag = build_model("agglomerative", {"params": ag_best, "n_clusters": n_clusters_ag})
            lab = ag.fit_predict(X)
            m = internal_scores(X, lab)
            board.append({
                "emb": emb,
                "algo": "agglomerative",
                "params": ag_best,
                "labels": lab,
                "metrics": m,
            })

        # ----- DBSCAN -----
        db_best = gridsearch_dbscan(X)
        if db_best and db_best.get("params"):
            board.append({
                "emb": emb,
                "algo": "dbscan",
                "params": db_best["params"],
                "labels": db_best["labels"],
                "metrics": db_best["metrics"],
            })

    return board


def rank_leaderboard(board):
    """Order candidates by your _score_internal (sil ↑, CH ↑, 1/(1+DB) ↑); tie-break: fewer clusters."""
    def key(row):
        m = row["metrics"]
        k = (m.get("n_clusters")
             or (row["params"].get("k") if row.get("params") else None)
             or 0)
        return (_score_internal(m), -int(k))  # prefer fewer clusters on ties
    return sorted(board, key=key, reverse=True)

def _safe_internal_scores(X, labels):
    """Compute internal metrics; return NaNs when invalid (e.g., 1 cluster)."""
    uniq = np.unique(labels[labels >= 0])  # ignore noise (if present)
    if len(uniq) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = np.nan
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

def _pick_best(df_metrics):
    """Rank rows and return the best index."""
    g = df_metrics.copy()
    g["r_sil"] = g["silhouette"].rank(ascending=False, method="min")
    g["r_ch"]  = g["calinski_harabasz"].rank(ascending=False, method="min")
    g["r_db"]  = g["davies_bouldin"].rank(ascending=True,  method="min")
    g["rank_sum"] = g[["r_sil","r_ch","r_db"]].sum(axis=1)

    g = g.dropna(subset=["rank_sum"])
    if g.empty: return None
    return g.sort_values("rank_sum").index[0]

def gridsearch_kmeans_params(X, ks=range(2, 11), n_inits=(10,25)):
    rows = []
    for k in ks:
        for n_init in n_inits:
            km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            labels = km.fit_predict(X)
            m = _safe_internal_scores(X, labels)
            rows.append({"k": k, "n_init": n_init, **m})
    df = pd.DataFrame(rows)
    idx = _pick_best(df)
    if idx is None:
        return {"params": None}
    best = df.loc[idx]
    return {"params": {"n_clusters": int(best["k"]), "n_init": int(best["n_init"])}}

def gridsearch_agglomerative_params(
    X,
    ks=range(2, 11),
    linkages=("ward","complete","average"),
    metrics=("euclidean","manhattan","cosine")
):
    rows = []
    for k in ks:
        for linkage in linkages:
            # ward requires euclidean
            metrics_to_try = ("euclidean",) if linkage == "ward" else metrics
            for metric in metrics_to_try:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
                labels = agg.fit_predict(X)
                m = _safe_internal_scores(X, labels)
                rows.append({"k": k, "linkage": linkage, "metric": metric, **m})
    df = pd.DataFrame(rows)
    idx = _pick_best(df)
    if idx is None:
        return {"params": None}
    best = df.loc[idx]
    return {"params": {"n_clusters": int(best["k"]), "linkage": best["linkage"], "metric": best["metric"]}}


def internal_scores(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Internal clustering scores z obsługą edge-case'ów:
    - n_clusters: liczba realnych klastrów (bez -1)
    - noise_%: udział punktów z etykietą -1
    - metryki liczone tylko, gdy są ≥2 realne klastry
    """
    labels = np.asarray(labels)
    uniq = set(labels)
    n_clusters = len([c for c in uniq if c != -1])
    noise_pct = float((labels == -1).mean() * 100.0)

    valid = (n_clusters >= 2)  # wymagamy ≥2 realnych klastrów (bez -1 jako „klastra”)
    def try_or_none(fn):
        try:
            return fn(X, labels) if valid else None
        except Exception:
            return None

    return {
        "silhouette": try_or_none(silhouette_score),
        "calinski_harabasz": try_or_none(calinski_harabasz_score),
        "davies_bouldin": try_or_none(davies_bouldin_score),
        "n_clusters": n_clusters,
        "noise_%": noise_pct,
    }
    
def _score_internal(m):
    # prefer higher silhouette, then higher CH, then lower DB
    sil = m.get("silhouette")
    ch  = m.get("calinski_harabasz")
    db  = m.get("davies_bouldin")

    bad = float("-inf")
    s_sil = float(sil) if sil is not None and not pd.isna(sil) else bad
    s_ch  = float(ch)  if ch  is not None and not pd.isna(ch)  else bad
    s_db  = float(db)  if db  is not None and not pd.isna(db)  else None

    db_part = (1.0 / (1.0 + s_db)) if s_db is not None else 0.0
    return (s_sil, s_ch, db_part)


def external_scores(labels: np.ndarray, y_true: Optional[pd.Series]) -> Dict[str, Optional[float]]:
    """Optional external metrics if ground truth available."""
    if y_true is None:
        return {"ARI": None, "NMI": None, "homogeneity": None, "completeness": None, "v_measure": None}
    try:
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        h, c, v = homogeneity_completeness_v_measure(y_true, labels)
        return {"ARI": ari, "NMI": nmi, "homogeneity": h, "completeness": c, "v_measure": v}
    except Exception:
        return {"ARI": None, "NMI": None, "homogeneity": None, "completeness": None, "v_measure": None}


def append_result(rows, embedding, algo, labels, scores_in, scores_ex, params=None):
    rows.append({
        "embedding": embedding,
        "algo": algo,
        "params": params,
        "n_clusters": scores_in.get("n_clusters"),
        "noise_%": scores_in.get("noise_%"),
        "silhouette": scores_in.get("silhouette"),
        "calinski_harabasz": scores_in.get("calinski_harabasz"),
        "davies_bouldin": scores_in.get("davies_bouldin"),
        "ARI": scores_ex.get("ARI") if scores_ex else None,
        "NMI": scores_ex.get("NMI") if scores_ex else None,
        "homogeneity": scores_ex.get("homogeneity") if scores_ex else None,
        "completeness": scores_ex.get("completeness") if scores_ex else None,
        "v_measure": scores_ex.get("v_measure") if scores_ex else None,
    })
    
def show_scores(name: str, X: np.ndarray, labels: np.ndarray, y_true=None):
    """
    Wyświetla metryki wewnętrzne i zewnętrzne dla klasteryzacji.
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


def plot_clusters_target_3d(X, cluster_labels, y_target, title="3D Clusters vs DEATH_EVENT"):
    """
    3D interactive scatter:
    - Color: DEATH_EVENT (blue=no death, orange=death)
    - Shape: cluster marker (circle, square, diamond...)
    """
    X = X.values if hasattr(X, "values") else np.asarray(X)
    cl = np.asarray(cluster_labels)
    yt = np.asarray(y_target)

    assert len(X) == len(cl) == len(yt)

    color_map = {0: "#0077BB", 1: "#EE7733"}  # DEATH_EVENT
    marker_cycle = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]
    unique_clusters = sorted(np.unique(cl))
    marker_map = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(unique_clusters)}

    fig = go.Figure()

    for c in unique_clusters:
        for t in [0, 1]:
            mask = (cl == c) & (yt == t)
            if np.any(mask):
                fig.add_trace(
                    go.Scatter3d(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        z=X[mask, 2],
                        mode="markers",
                        marker=dict(size=5, color=color_map[t], symbol=marker_map[c], line=dict(color="white", width=0.8), opacity=0.8),
                        name=f"Cluster {c} – {'Death' if t else 'No death'}",
                        hovertemplate=(f"Cluster: {c}<br>DEATH_EVENT: {'Death' if t else 'No'}"+ "<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"),
                    )
                )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Component 1", yaxis_title="Component 2", zaxis_title="Component 3"),
        legend=dict(bgcolor="white", bordercolor="lightgray", borderwidth=1),
        height=700,
        width=900,
    )

    fig.show()


def cluster_profile(df_original: pd.DataFrame, labels: np.ndarray, top_n: int = 8):
    df = df_original.copy()
    df["_cluster"] = labels

    num_cols = df.select_dtypes(include="number").columns.drop("_cluster", errors="ignore")

    prof = df.groupby("_cluster")[num_cols].agg(['mean','median','std'])
    overall = df[num_cols].agg(['mean','median','std'])
    return prof, overall


def k_distance_plot(X, k=5):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    kd = np.sort(dists[:, -1])
    return kd

def gridsearch_dbscan(X, eps_grid=(0.2,0.3,0.5,0.8,1.2), min_samples_grid=(3,5,10), scorer="silhouette"):
    best = None
    for eps in eps_grid:
        for ms in min_samples_grid:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            m = internal_scores(X, labels)
            score = m["silhouette"] if scorer=="silhouette" else -m["davies_bouldin"] if m["davies_bouldin"] is not None else None
            if score is None: 
                continue
            cand = dict(params={"eps":eps,"min_samples":ms}, metrics=m, labels=labels)
            if best is None or score > (best["metrics"]["silhouette"] if scorer=="silhouette" else -best["metrics"]["davies_bouldin"]):
                best = cand
    return best

def _compute_internal_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    sil = silhouette_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

def _plot_curves(df: pd.DataFrame, title: str, include_inertia: bool = False, tag: str | None = None) -> None:
    """Plot metric curves vs k and optionally stamp a dataset tag (e.g., 'PCA', 'UMAP')."""
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    if include_inertia and "inertia" in df.columns:
        metrics = ["inertia"] + metrics

    fig = plt.figure(figsize=(9, 8))
    for i, m in enumerate(metrics, start=1):
        plt.subplot(len(metrics), 1, i)
        plt.plot(df["k"], df[m], marker="o")
        plt.xlabel("k"); plt.ylabel(m.replace("_", " ").title())
        plt.grid(True)
        if i == 1:
            plt.title(title)

    fig.text(0.99, 0.995, f"DATA: {tag}", ha="right", va="top", bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    plt.tight_layout()
    plt.show()

@dataclass
class SelectionResult:
    df_scores: pd.DataFrame
    best_by_metric: Dict[str, int]   # metric -> k
    extra: Optional[Dict] = None     # e.g., linkage used, additional info


def select_k_kmeans(X, k_range=range(2, 21), n_init=range(10,25), random_state=42, data_name: str | None = None) -> SelectionResult:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        met = _compute_internal_metrics(X, labels)
        rows.append({"k": k, "inertia": float(km.inertia_), **met})

    df = pd.DataFrame(rows)
    best = {
        "inertia":            int(df.loc[df["inertia"].idxmin(), "k"]),
        "silhouette":         int(df.loc[df["silhouette"].idxmax(), "k"]),
        "calinski_harabasz":  int(df.loc[df["calinski_harabasz"].idxmax(), "k"]),
        "davies_bouldin":     int(df.loc[df["davies_bouldin"].idxmin(), "k"]),
    }

    _plot_curves(df, title="KMeans model selection", include_inertia=True, tag=(data_name.upper() if data_name else None))
    return SelectionResult(df_scores=df, best_by_metric=best)


def select_k_agglomerative(X, k_range=range(2, 21), linkage: str = "ward", metric: str | None = None, data_name: str | None = None) -> SelectionResult:
    rows = []
    for k in k_range:
        agg = (AgglomerativeClustering(n_clusters=k, linkage="ward")
               if linkage == "ward"
               else AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric or "euclidean"))
        labels = agg.fit_predict(X)
        met = _compute_internal_metrics(X, labels)
        rows.append({"k": k, **met})

    df = pd.DataFrame(rows)
    best = {
        "silhouette":        int(df.loc[df["silhouette"].idxmax(), "k"]),
        "calinski_harabasz": int(df.loc[df["calinski_harabasz"].idxmax(), "k"]),
        "davies_bouldin":    int(df.loc[df["davies_bouldin"].idxmin(), "k"]),
    }

    _plot_curves(df, title=f"Agglomerative model selection (linkage={linkage}{', metric='+metric if metric else ''})", include_inertia=False, tag=(data_name.upper() if data_name else None))
    return SelectionResult(df_scores=df, best_by_metric=best, extra={"linkage": linkage, "metric": metric or ("euclidean" if linkage != "ward" else "euclidean")})


def build_model(algo: str, w: dict):
    p = w.get("params", {}) or {}
    k = int(w.get("n_clusters", p.get("k", p.get("n_clusters", 2))))

    if algo == "dbscan":
        return DBSCAN(**p)

    if algo == "kmeans":
        extra = {kk: vv for kk, vv in p.items() if kk not in {"k", "n_clusters"}}
        return KMeans(n_clusters=k, random_state=42, **extra)

    if algo == "agglomerative":
        linkage = p.get("linkage", "ward")
        metric  = p.get("metric", "euclidean")
        kwargs = {"n_clusters": k, "linkage": linkage}
        if linkage != "ward":
            kwargs["metric"] = metric
        return AgglomerativeClustering(**kwargs)

    raise ValueError(f"Unknown algo: {algo}")

def compute_smds(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMDs) per feature × cluster.
    |SMD| = |mean_cluster - mean_all| / std_all
    """
    labels = np.asarray(labels)
    if len(labels) != len(X):
        raise ValueError(f"labels length {len(labels)} != X length {len(X)}")

    Xn = X.select_dtypes(include="number").copy()
    df = Xn.copy()
    df["_cluster"] = labels

    mean_all = Xn.mean()
    std_all  = Xn.std(ddof=0).replace(0, np.finfo(float).eps)

    smd_df = pd.DataFrame(index=Xn.columns)
    for c, g in df.groupby("_cluster"):
        mean_c = g.drop(columns="_cluster").mean()
        smd_df[f"Cluster {c}"] = (mean_c - mean_all) / std_all

    return smd_df

def top_union_smd_table(smd_df: pd.DataFrame, top_n: int = 8, feature_order: str = "max"):
    """
    Create one wide table:
      - rows: features (each appears once)
      - columns: Cluster 0..K with |SMD| values
      - features = union of top-N |SMD| from each cluster
      - feature order:
          "max"  -> by descending max |SMD| across clusters
          "sum"  -> by descending sum |SMD| across clusters
          "alpha"-> alphabetical by feature name
          None   -> no reordering (original index order)
    """
    # 1) union of top-N features per cluster
    top_sets = []
    for c in smd_df.columns:
        top_feats = smd_df[c].abs().nlargest(top_n).index
        top_sets.append(set(top_feats))
    all_feats = sorted(set().union(*top_sets))  # de-duplicate

    # 2) build matrix Feature x Cluster with |SMD|
    out = pd.DataFrame(index=all_feats)
    for c in smd_df.columns:  # keep original cluster column order
        col = smd_df[c].abs().reindex(all_feats)
        # keep values only for that cluster's top-N, else NaN
        top_mask = smd_df[c].abs().rank(ascending=False, method="first") <= top_n
        allowed = set(smd_df.index[top_mask])
        col = col.where(out.index.to_series().isin(allowed))
        out[c] = col

    # 3) choose feature order
    if feature_order == "max":
        order = out.max(axis=1).sort_values(ascending=False).index
    elif feature_order == "sum":
        order = out.sum(axis=1, skipna=True).sort_values(ascending=False).index
    elif feature_order == "alpha":
        order = sorted(out.index)
    else:
        order = out.index  # as-is

    out = out.loc[order]
    out.index.name = "Feature"
    return out

def plot_smd_stacked_bars(
    smd_df: pd.DataFrame,
    rank: int,
    emb: str,
    algo: str,
    top_n: int = 8,
):
    top_feats = set()
    for col in smd_df.columns:
        top_feats |= set(smd_df[col].abs().nlargest(top_n).index)
    top_feats = list(top_feats)

    df_plot = smd_df.loc[top_feats].copy()
    df_plot = df_plot.abs().fillna(0)
    df_plot["Total"] = df_plot.sum(axis=1)
    df_plot = df_plot.sort_values("Total").drop(columns="Total")

    df_plot.plot.barh(stacked=True)
    plt.xlabel("|Standardized Mean Difference|")
    plt.ylabel("")
    plt.title(f"[{rank}] Stacked |SMD|s | {emb} – {algo}", pad=10)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
    
def plot_smd_heatmap(
    smd_df: pd.DataFrame,
    rank: int,
    emb: str,
    algo: str,
):
    sns.heatmap(smd_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"[{rank}] Top SMDs by Cluster | {emb} – {algo}")
    plt.ylabel("")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()