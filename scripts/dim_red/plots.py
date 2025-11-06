import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_cumulative_variance(explained_variance, show_target: float | None = 0.95):
    """
    Display a cumulative explained variance plot directly below the notebook cell.

    Parameters
    ----------
    explained_variance : array-like
        The PCA explained_variance_ratio_ array.
    show_target : float or None, optional
        If set (e.g. 0.95), draws a horizontal line at that cumulative variance.
    """
    cumulative = np.cumsum(explained_variance)
    components = np.arange(1, len(explained_variance) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(components, np.cumsum(explained_variance), marker='o', color='blue')
    plt.xlabel("Number of Components")
    plt.xticks(components)
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True, linestyle='--', alpha=0.5)

    if show_target is not None:
        plt.axhline(show_target, color='red', linestyle='--', linewidth=1)
        plt.text(
            len(cumulative) * 0.7,
            show_target - 0.05,
            f"{show_target:.0%} variance",
            color='red',
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_3d(
    X3d: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    algorithm: str = "PCA",
    target_name: str = "num",
    data_name: str | None = None,
    s: int = 5,
    alpha: float = 0.8,
    show: bool = False,
):
    """
    Interactive 3D scatter (Plotly) for dimensionality reduction results.

    Features:
    - One interactive plot (no repeated plots per angle)
    - Clickable legend for each class
    - Hover tooltips showing coordinates and class
    """

    X = X3d.values if isinstance(X3d, pd.DataFrame) else np.asarray(X3d)
    labels_xyz = ["Component 1", "Component 2", "Component 3"]

    y = pd.Series(y).reset_index(drop=True)
    classes = pd.Index(sorted(y.unique(), key=lambda v: str(v)))

    fig = go.Figure()
    for i, c in enumerate(classes):
        mask = (y == c).to_numpy()
        fig.add_trace(
            go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2],
                mode="markers",
                name=str(c),
                marker=dict(size=s, opacity=alpha, line=dict(width=0)),
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
        scene=dict(xaxis_title=labels_xyz[0], yaxis_title=labels_xyz[1], zaxis_title=labels_xyz[2]),
        legend=dict(itemsizing="trace", title=dict(text=target_name)),
    )

    if data_name:
        fig.add_annotation(
            text=f"<b>DATA:</b> {data_name.upper()}",
            showarrow=False,
            xref="paper", yref="paper",
            x=1, y=1,
            xanchor="right", yanchor="top",
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            opacity=0.8,
        )

    if show: fig.show()
    return fig