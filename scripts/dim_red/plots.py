import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


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

    plt.figure(figsize=(7, 4))
    components = np.arange(1, len(explained_variance) + 1)
    plt.plot(components, np.cumsum(explained_variance), marker='o', color='blue')
    plt.xlabel("Number of Components")
    plt.xticks(components)  # show 1–12 ticks
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
    
    
def plot(
    X3d: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    algorithm: str = "PCA",
    target_name: str = "num",
    angles: tuple[tuple[int, int], ...] = ((90, 90), (0, 90), (0, 0)),
    figsize: tuple[int, int] = (6, 6),
    s: int = 20,
    alpha: float = 0.6,
) -> None:
    """
    Plot 3D scatter of first three components with multiple viewing angles.

    Parameters
    ----------
    X3d : DataFrame or ndarray, shape (n_samples, 3)
        3D embedding (e.g., first 3 PCA components). If DataFrame, columns may be
        ['PC1','PC2','PC3']; otherwise they will be labeled generically.
    y : array-like, shape (n_samples,)
        Target labels (binary or multiclass). Can be ints or strings.
    algorithm : str
        Name shown in the plot title (e.g., "PCA", "t-SNE").
    target_name : str
        Name used in the legend (e.g., "DEATH_EVENT").
    angles : tuple of (elev, azim)
        Camera angles to render. One figure is created per angle.
    figsize : tuple
        Figure size in inches.
    s : int
        Marker size.
    alpha : float
        Marker opacity.
    """
    # --- Prepare inputs
    X = X3d.values if isinstance(X3d, pd.DataFrame) else np.asarray(X3d)
    if X.shape[1] != 3:
        raise ValueError("X3d must have exactly 3 columns/components.")

    if isinstance(X3d, pd.DataFrame):
        labels_xyz = list(X3d.columns)
    else:
        labels_xyz = ["Component 1", "Component 2", "Component 3"]

    y = pd.Series(y).reset_index(drop=True)
    classes = pd.Index(sorted(y.unique(), key=lambda v: str(v)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y.map(class_to_idx).to_numpy()

    # Use a discrete colormap with as many colors as classes
    cmap = plt.get_cmap("viridis", len(classes))

    # --- Draw a figure per requested angle
    for elev, azim in angles:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=y_idx,
            cmap=cmap,
            s=s,
            alpha=alpha,
            depthshade=True,
        )

        ax.set_xlabel(labels_xyz[0])
        ax.set_ylabel(labels_xyz[1])
        ax.set_zlabel(labels_xyz[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"3D {algorithm} with {target_name} angles=({elev},{azim})")

        # Build legend with class labels
        handles, _ = sc.legend_elements(num=len(classes))
        legend = ax.legend(handles, [str(c) for c in classes], title=target_name, loc="upper right")
        for lh in legend.legend_handles:
            lh.set_alpha(1.0)

        plt.tight_layout()
        plt.show()
        
    
# def plot_3d(
#     X3d: pd.DataFrame | np.ndarray,
#     y: pd.Series | np.ndarray,
#     algorithm: str = "PCA",
#     target_name: str = "num",
#     angles: tuple[tuple[int, int], ...] = ((30, 45), (60, 120), (0, 180)),
#     s: int = 5,
#     alpha: float = 0.8,
#     height: int = 650,
#     width: int = 800,
#     show: bool = False,
# ):
#     """
#     Interactive 3D scatter (Plotly) for dimensionality reduction results.

#     Features:
#     - One interactive plot (no repeated plots per angle)
#     - Clickable legend for each class
#     - Hover tooltips showing coordinates and class
#     - Optional dropdown to switch to predefined camera angles
#     """

#     # --- Prepare inputs
#     X = X3d.values if isinstance(X3d, pd.DataFrame) else np.asarray(X3d)
#     if X.ndim != 2 or X.shape[1] != 3:
#         raise ValueError("X3d must have shape (n_samples, 3).")

#     if isinstance(X3d, pd.DataFrame):
#         labels_xyz = list(X3d.columns)
#     else:
#         labels_xyz = ["Component 1", "Component 2", "Component 3"]

#     y = pd.Series(y).reset_index(drop=True)
#     classes = pd.Index(sorted(y.unique(), key=lambda v: str(v)))

#     # Use a qualitative palette for distinct class colors
#     palette = qualitative.Plotly + qualitative.D3 + qualitative.Light24
#     if len(classes) > len(palette):
#         palette = (palette * ((len(classes) // len(palette)) + 1))[: len(classes)]

#     # --- Create the interactive 3D scatter
#     fig = go.Figure()

#     for i, c in enumerate(classes):
#         mask = (y == c).to_numpy()
#         fig.add_trace(
#             go.Scatter3d(
#                 x=X[mask, 0],
#                 y=X[mask, 1],
#                 z=X[mask, 2],
#                 mode="markers",
#                 name=str(c),
#                 marker=dict(size=s, opacity=alpha, color=palette[i], line=dict(width=0)),
#                 hovertemplate=(
#                     f"<b>{target_name}</b>: {str(c)}<br>"
#                     f"{labels_xyz[0]}: %{{x:.3f}}<br>"
#                     f"{labels_xyz[1]}: %{{y:.3f}}<br>"
#                     f"{labels_xyz[2]}: %{{z:.3f}}<extra></extra>"
#                 ),
#             )
#         )

#     # --- Helper to convert (elev, azim) → Plotly camera eye vector
#     def eye_from_elev_azim(elev_deg: float, azim_deg: float, r: float = 2.2):
#         elev, azim = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
#         x = r * np.cos(elev) * np.cos(azim)
#         y = r * np.cos(elev) * np.sin(azim)
#         z = r * np.sin(elev)
#         return dict(x=x, y=y, z=z)

#     # --- Dropdown for camera angle presets
#     buttons = [
#         dict(
#             label=f"View {i+1} (elev={e}, azim={a})",
#             method="relayout",
#             args=[{"scene.camera": {"eye": eye_from_elev_azim(e, a)}}],
#         )
#         for i, (e, a) in enumerate(angles)
#     ]

#     fig.update_layout(
#         title=f"3D {algorithm} colored by {target_name}",
#         width=width,
#         height=height,
#         scene=dict(
#             xaxis_title=labels_xyz[0],
#             yaxis_title=labels_xyz[1],
#             zaxis_title=labels_xyz[2],
#             camera=dict(eye=eye_from_elev_azim(*angles[0])),
#         ),
#         legend=dict(itemsizing="trace"),
#         margin=dict(l=0, r=0, t=60, b=0),
#         updatemenus=[
#             dict(
#                 type="dropdown",
#                 buttons=buttons,
#                 x=0.01,
#                 y=1.08,
#                 xanchor="left",
#                 yanchor="top",
#                 showactive=True,
#             )
#         ],
#     )

#     if show:
#         fig.show()
#     return fig


def plot_3d(
    X3d: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    algorithm: str = "PCA",
    target_name: str = "num",
    angles: tuple[tuple[int, int], ...] = ((30, 45), (60, 120), (0, 180)),
    s: int = 5,
    alpha: float = 0.8,
    height: int = 650,
    width: int = 800,
    show: bool = False,
):
    """
    Interactive 3D scatter (Plotly) for dimensionality reduction results.

    Features:
    - One interactive plot (no repeated plots per angle)
    - Clickable legend for each class
    - Hover tooltips showing coordinates and class
    - Optional dropdown to switch to predefined camera angles
    """

    # Prepare inputs
    X = X3d.values if isinstance(X3d, pd.DataFrame) else np.asarray(X3d)
    if np.isnan(X).any():
        raise ValueError("X3d zawiera NaN – oczyść dane przed wizualizacją.")
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("Potrzebne co najmniej 2 wymiary do 3D (2D zostanie rozszerzone do 3D).")
    if X.shape[1] == 2:
        X = np.c_[X, np.zeros((X.shape[0], 1))]
    elif X.shape[1] > 3:
        X = X[:, :3]

    if isinstance(X3d, pd.DataFrame) and X3d.shape[1] >= 3:
        labels_xyz = list(X3d.columns[:3])
    else:
        labels_xyz = ["Component 1", "Component 2", "Component 3"]

    y = pd.Series(y).reset_index(drop=True)
    classes = pd.Index(sorted(y.unique(), key=lambda v: str(v)))

    # palette
    palette = qualitative.Plotly + qualitative.D3 + qualitative.Light24
    if len(classes) > len(palette):
        times = (len(classes) // len(palette)) + 1
        palette = (palette * times)[: len(classes)]

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
                marker=dict(size=s, opacity=alpha, color=palette[i], line=dict(width=0)),
                hovertemplate=(
                    f"<b>{target_name}</b>: {str(c)}<br>"
                    f"{labels_xyz[0]}: %{{x:.3f}}<br>"
                    f"{labels_xyz[1]}: %{{y:.3f}}<br>"
                    f"{labels_xyz[2]}: %{{z:.3f}}<extra></extra>"
                ),
            )
        )

    def eye_from_elev_azim(elev_deg: float, azim_deg: float, r: float = 2.2):
        elev, azim = np.deg2rad(elev_deg), np.deg2rad(azim_deg)
        return dict(
            x=r * np.cos(elev) * np.cos(azim),
            y=r * np.cos(elev) * np.sin(azim),
            z=r * np.sin(elev),
        )

    buttons = [
        dict(
            label=f"View {i+1} (elev={e}, azim={a})",
            method="relayout",
            args=[{"scene.camera": {"eye": eye_from_elev_azim(e, a)}}],
        )
        for i, (e, a) in enumerate(angles)
    ]

    fig.update_layout(
        title=f"3D {algorithm} colored by {target_name}",
        width=width,
        height=height,
        scene=dict(
            xaxis_title=labels_xyz[0],
            yaxis_title=labels_xyz[1],
            zaxis_title=labels_xyz[2],
            camera=dict(eye=eye_from_elev_azim(*angles[0])),
            aspectmode="data",  # równe skale osi
        ),
        legend=dict(itemsizing="trace", title=dict(text=target_name)),
        margin=dict(l=0, r=0, t=60, b=0),
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.01, y=1.08, xanchor="left", yanchor="top", showactive=True)],
        uirevision="keep_view",  # nie resetuj kamery po aktualizacjach
    )

    if show:
        fig.show()
    return fig