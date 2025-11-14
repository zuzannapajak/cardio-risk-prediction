import numpy as np, pandas as pd, matplotlib.pyplot as plt, plotly.graph_objects as go
from typing import Any, Optional, Sequence
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import ClassifierMixin


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _to_numpy(x: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def plot_cumulative_variance(
    explained_variance: Sequence[float],
    show_target: float | None = 0.95,
) -> None:
    """
    Plot cumulative explained variance from PCA.
    """
    ev = np.asarray(explained_variance, dtype=float)
    cumulative = np.cumsum(ev)
    components = np.arange(1, len(ev) + 1)

    plt.figure(figsize=(7,4))
    plt.plot(components, cumulative, marker="o")
    plt.xlabel("Number of Components")
    plt.xticks(components)
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axhline(show_target, linestyle="--", linewidth=1)
    plt.text(
        max(1, int(len(cumulative) * 0.7)),
        show_target - 0.05,
        f"{show_target:.0%} variance",
        fontsize=9,
    )

    plt.tight_layout()
    plt.show()


def plot_3d(
    X3d: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    algorithm: str = "PCA",
    target_name: str = "DEATH_EVENT",
    data_name: str | None = None,
    marker_size: int = 5,
    marker_opacity: float = 0.8,
    show: bool = False,
) -> go.Figure:
    """
    Interactive 3D scatter (Plotly) for dimensionality reduction results.
    """
    X = _to_numpy(X3d)
    y_series = pd.Series(_to_numpy(y)).reset_index(drop=True)
    
    classes = pd.Index(sorted(y_series.unique(), key=lambda v: str(v)))
    labels_xyz = ["Component 1", "Component 2", "Component 3"]

    fig = go.Figure()
    for c in classes:
        mask = (y_series == c).to_numpy()
        if not np.any(mask):
            continue
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode="markers",
                name=str(c),
                marker=dict(size=marker_size, opacity=marker_opacity, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{target_name}</b>: {str(c)}<br>"
                    f"{labels_xyz[0]}: %{{x:.3f}}<br>"
                    f"{labels_xyz[1]}: %{{y:.3f}}<br>"
                    f"{labels_xyz[2]}: %{{z:.3f}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"3D {algorithm} colored by {target_name}",
        height=700,
        width=900,
        scene=dict(xaxis_title=labels_xyz[0], yaxis_title=labels_xyz[1], zaxis_title=labels_xyz[2],),
        legend=dict(itemsizing="trace", title=dict(text=target_name)),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    if data_name:
        fig.add_annotation(
            text=f"<b>DATA:</b> {str(data_name).upper()}",
            showarrow=False,
            xref="paper", yref="paper",
            x=1, y=1,
            xanchor="right", yanchor="top",
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8,
        )

    if show:
        fig.show()
    return fig


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def eval_embedding(
    name: str,
    Z: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    X_ref: Optional[np.ndarray | pd.DataFrame] = None,
    *,
    k: int = 10,
    cv_splits: int = 5,
    clf: Optional[ClassifierMixin] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Evaluate a 2D/3D embedding.

    Metrics
    -------
    - Trustworthiness@k (if X_ref provided)
    - CV ROC-AUC of a simple classifier trained on Z (default: LogisticRegression)
    """
    Z_np = _to_numpy(Z)
    y_np = _to_numpy(y).ravel()

    # Classifier for linear separability probe
    clf = clf or LogisticRegression(max_iter=5000, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_auc = cross_val_score(clf, Z_np, y_np, cv=cv, scoring="roc_auc").mean()

    out: dict[str, Any] = {
        "Embedding": name,
        "Classifier": clf.__class__.__name__,
        "CV ROC-AUC": float(cv_auc),
    }

    if X_ref is not None:
        X_np = _to_numpy(X_ref)
        k_eff = min(k, max(1, len(X_np) - 1))
        out[f"Trustworthiness@{k_eff}"] = float(trustworthiness(X_np, Z_np, n_neighbors=k_eff))
    return out
