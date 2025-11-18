from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_smd_stacked_bars(
    smd_df: pd.DataFrame,
    rank: int,
    emb: str,
    algo: str,
    top_n: int = 8,
) -> None:
    """
    Plot stacked horizontal bars of |SMD| values for top features.

    Parameters
    ----------
    smd_df : DataFrame
        Output of compute_smds(...).
    rank : int
        Rank index (e.g. from leaderboard) to display in the title.
    emb : str
        Embedding name (e.g. 'PCA').
    algo : str
        Algorithm name (e.g. 'KMeans').
    top_n : int
        Number of top features per cluster included in the union.
    """
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
) -> None:
    """
    Plot a heatmap of SMD values (or |SMD| if you take abs before passing).

    Parameters
    ----------
    smd_df : DataFrame
        Typically a filtered/ordered SMD table (e.g. top_union_smd_table).
    rank : int
        Rank index (e.g. from leaderboard).
    emb : str
        Embedding name.
    algo : str
        Algorithm name.
    """
    sns.heatmap(smd_df, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title(f"[{rank}] Top SMDs by Cluster | {emb} – {algo}")
    plt.ylabel("")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()
