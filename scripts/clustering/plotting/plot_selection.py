from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_k_selection_curves(
    df: pd.DataFrame,
    title: str,
    include_inertia: bool = False,
    tag: Optional[str] = None,
) -> None:
    """
    Plot internal metrics (and optionally inertia) vs k.

    Parameters
    ----------
    df : DataFrame
        Must contain column 'k' and metric columns:
        - silhouette
        - calinski_harabasz
        - davies_bouldin
        - optionally 'inertia'
    title : str
        Title of the figure.
    include_inertia : bool
        If True, plot inertia as the first curve (if present).
    tag : Optional[str]
        Optional dataset tag stamped in the upper-right corner.
    """
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    if include_inertia and "inertia" in df.columns:
        metrics = ["inertia"] + metrics

    fig = plt.figure(figsize=(9, 8))
    for i, m in enumerate(metrics, start=1):
        plt.subplot(len(metrics), 1, i)
        plt.plot(df["k"], df[m], marker="o")
        plt.xlabel("k")
        plt.ylabel(m.replace("_", " ").title())
        plt.grid(True)
        if i == 1:
            plt.title(title)

    if tag:
        fig.text(
            0.99,
            0.995,
            f"DATA: {tag}",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9),
        )

    plt.tight_layout()
    plt.show()
