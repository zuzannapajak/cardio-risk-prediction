from typing import Dict, Optional
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score,
)

def _align_classes(y_true, proba, classes: Optional[np.ndarray]) -> np.ndarray:
    """
    Returns the class order to use for alignment.
    If classes is None, default to sorted unique y_true.
    """
    if classes is None:
        classes = np.unique(y_true)
    # sanity: proba columns must match the number of classes in multiclass
    if proba.ndim == 2 and proba.shape[1] > 1 and proba.shape[1] != len(classes):
        raise ValueError("proba.shape[1] does not match number of classes.")
    return np.asarray(classes)

def _pick_pos_proba(y_true, proba, classes: Optional[np.ndarray]) -> np.ndarray:
    """Return p(pos) aligned to label 1 if present; otherwise use column 1."""
    if proba.ndim == 1 or proba.shape[1] == 1:
        return proba.ravel()
    classes = _align_classes(y_true, proba, classes)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = 1  # fallback
    return proba[:, pos_idx]

def _brier_multiclass(y_true, proba, classes) -> float:
    Y = label_binarize(y_true, classes=classes)
    return float(((Y - proba) ** 2).mean())

def eval_probs(y_true, proba, classes: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Evaluate probability predictions.
    - Binary: ROC-AUC, PR-AUC, Brier, BalancedAcc@0.5 (thresholded)
    - Multiclass: ROC-AUC (macro OvR), PR-AUC (macro OvR), Brier (multiclass), BalancedAcc (argmax)
    """
    out: Dict[str, float] = {}

    n_classes = 2 if proba.ndim == 1 else proba.shape[1]
    unique_y = np.unique(y_true)
    classes = _align_classes(y_true, proba, classes)

    if n_classes <= 2 and len(unique_y) <= 2:
        p_pos = _pick_pos_proba(y_true, proba, classes)
        y_pred = (p_pos >= 0.5).astype(int)

        out["ROC-AUC"] = float(roc_auc_score(y_true, p_pos))
        out["PR-AUC"] = float(average_precision_score(y_true, p_pos))
        out["Brier"] = float(brier_score_loss(y_true, p_pos))
        out["BalancedAcc@0.5"] = float(balanced_accuracy_score(y_true, y_pred))
        return out

    # Multiclass
    out["ROC-AUC(macro-ovr)"] = float(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    )
    Y = label_binarize(y_true, classes=classes)
    out["PR-AUC(macro-ovr)"] = float(average_precision_score(Y, proba, average="macro"))
    out["Brier"] = _brier_multiclass(y_true, proba, classes=classes)

    hard_idx = np.argmax(proba, axis=1)
    if len(classes) == proba.shape[1]:
        y_pred = classes[hard_idx]
    else:
        y_pred = hard_idx
    out["BalancedAcc"] = float(balanced_accuracy_score(y_true, y_pred))
    return out
