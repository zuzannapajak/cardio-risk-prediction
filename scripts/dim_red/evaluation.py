import numpy as np, pandas as pd
from typing import Any, Optional
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from .utils import to_numpy


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
    Z_np = to_numpy(Z)
    y_np = to_numpy(y).ravel()

    clf = clf or LogisticRegression(max_iter=5000, random_state=random_state)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    cv_auc = cross_val_score(clf, Z_np, y_np, cv=cv, scoring="roc_auc").mean()

    out: dict[str, Any] = {
        "Embedding": name,
        "Classifier": clf.__class__.__name__,
        "CV ROC-AUC": float(cv_auc),
    }

    if X_ref is not None:
        X_np = to_numpy(X_ref)
        k_eff = min(k, max(1, len(X_np) - 1))
        out[f"Trustworthiness@{k_eff}"] = float(
            trustworthiness(X_np, Z_np, n_neighbors=k_eff)
        )

    return out
