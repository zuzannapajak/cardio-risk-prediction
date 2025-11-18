from typing import Any

import numpy as np
import plotly.graph_objects as go


def plot_clusters_target_3d(
    X,
    cluster_labels,
    y_target,
    title: str = "3D Clusters vs Target",
    target_name: str = "DEATH_EVENT",
):
    """
    3D interactive plot showing clusters vs target labels.

    - Color encodes the target (e.g. 0 = no event, 1 = event).
    - Marker shape encodes cluster membership.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_samples, 3)
        3D embedding coordinates.
    cluster_labels : array-like
        Cluster assignments.
    y_target : array-like
        Target labels (binary or categorical).
    title : str
        Plot title.
    target_name : str
        Name of the target variable for legend/hover text.
    """
    X = X.values if hasattr(X, "values") else np.asarray(X)
    cl = np.asarray(cluster_labels)
    yt = np.asarray(y_target)

    assert len(X) == len(cl) == len(yt)

    color_map = { 0: "#0077BB", 1:"#EE7733"}
    marker_cycle = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]
    unique_clusters = sorted(np.unique(cl))
    marker_map = {c: marker_cycle[i % len(marker_cycle)] for i, c in enumerate(unique_clusters)}

    fig = go.Figure()

    unique_targets = sorted(np.unique(yt))
    for c in unique_clusters:
        for t in unique_targets:
            mask = (cl == c) & (yt == t)
            if np.any(mask): color = color_map.get(int(t), "#555555")
            fig.add_trace(
                go.Scatter3d(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    z=X[mask, 2],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.8,
                        symbol=marker_map[c],
                        line=dict(color="white", width=0.8),
                    ),
                    name=f"Cluster {c} – {target_name}={t}",
                    hovertemplate=(
                        f"Cluster: {c}<br>{target_name}: {t}"
                        "<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Component 1", yaxis_title="Component 2", zaxis_title="Component 3",),
        legend=dict(bgcolor="white", bordercolor="lightgray", borderwidth=1),
        height=700,
        width=900,
    )

    fig.show()
