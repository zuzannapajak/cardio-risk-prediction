import numpy as np
import pandas as pd


def compute_smds(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Compute standardized mean differences (SMDs) per feature × cluster.

    SMD = (mean_cluster - mean_overall) / std_overall

    Returns
    -------
    DataFrame
        index: feature names
        columns: 'Cluster <id>'
    """
    labels = np.asarray(labels)
    if len(labels) != len(X):
        raise ValueError(f"labels length {len(labels)} != X length {len(X)}")

    Xn = X.select_dtypes(include="number").copy()
    df = Xn.copy()
    df["_cluster"] = labels

    mean_all = Xn.mean()
    std_all = Xn.std(ddof=0).replace(0, np.finfo(float).eps)

    smd_df = pd.DataFrame(index=Xn.columns)
    for c, g in df.groupby("_cluster"):
        mean_c = g.drop(columns="_cluster").mean()
        smd_df[f"Cluster {c}"] = (mean_c - mean_all) / std_all

    return smd_df


def top_union_smd_table(
    smd_df: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Build a compact |SMD| summary table across clusters.

    Steps:
    - For each cluster, take top-N features by |SMD|
    - Take the union of those features across all clusters (rows)
    - For each cluster, keep |SMD| only for its own top-N, others -> NaN

    Returns
    -------
    DataFrame
        Index = selected features
        Columns = 'Cluster ...' with |SMD| values (NaN for non-top-N)
    """
    top_sets = []
    for c in smd_df.columns:
        top_feats = smd_df[c].abs().nlargest(top_n).index
        top_sets.append(set(top_feats))
    all_feats = sorted(set().union(*top_sets))  # de-duplicate

    out = pd.DataFrame(index=all_feats)
    for c in smd_df.columns:  # keep original cluster column order
        col = smd_df[c].abs().reindex(all_feats)

        top_mask = smd_df[c].abs().rank(ascending=False, method="first") <= top_n
        allowed = set(smd_df.index[top_mask])
        col = col.where(out.index.to_series().isin(allowed))
        out[c] = col

    order = out.max(axis=1).sort_values(ascending=False).index

    out = out.loc[order]
    out.index.name = "Feature"
    return out
