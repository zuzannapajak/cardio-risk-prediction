import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_oof_probability_distributions(
    y_true,
    y_proba,
    model_name: str,
    threshold: float,
    bins: int = 40,
    save_path: str | None = None
):
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df[df["y_true"] == 0],
        x="y_proba",
        bins=bins,
        kde=True,
        stat="density",
        label="DEATH_EVENT = 0",
        alpha=0.45
    )
    sns.histplot(
        data=df[df["y_true"] == 1],
        x="y_proba",
        bins=bins,
        kde=True,
        stat="density",
        label="DEATH_EVENT = 1",
        alpha=0.45
    )

    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    plt.title(f"OOF probability distributions – {model_name}")
    plt.xlabel("Predicted probability of DEATH_EVENT = 1 (OOF)")
    plt.ylabel("Density")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()