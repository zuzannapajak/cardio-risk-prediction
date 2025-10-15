import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    
    
def plot_3d(
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