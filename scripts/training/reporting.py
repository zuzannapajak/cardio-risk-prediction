from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from .metrics import eval_probs, _pick_pos_proba

def print_eval_report(
    y_true,
    proba,
    classes: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    verbose: bool = True,  # NEW: control printing
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Pretty, universal report:
      - Probability metrics (via eval_probs)
      - Hard-pred metrics (+ confusion matrix & classification report)

    Returns:
        (prob_metrics, hard_metrics) as dictionaries.

    Notes:
    - Binary hard preds use 'threshold' on the positive-class probability.
    - Multiclass hard preds use argmax.
    - 'verbose=False' suppresses printing (useful in notebooks to avoid duplicate outputs).
    """

    # --- Probability metrics ---
    prob_metrics = eval_probs(y_true, proba, classes=classes)

    # --- Hard predictions ---
    if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] <= 2 and len(np.unique(y_true)) <= 2):
        # Binary
        p_pos = _pick_pos_proba(y_true, proba, classes)
        y_pred = (p_pos >= threshold).astype(int)
    else:
        # Multiclass
        hard_idx = np.argmax(proba, axis=1)
        if classes is not None and len(classes) == proba.shape[1]:
            y_pred = np.asarray(classes)[hard_idx]
        else:
            y_pred = hard_idx

    hard = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1-macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    if verbose:
        print("=== Probability metrics ===")
        for k, v in prob_metrics.items():
            print(f"{k}: {v:.3f}")
        print()

        print("=== Hard-pred metrics ===")
        for k, v in hard.items():
            print(f"{k}: {v:.3f}")
        print()

        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print()

        print("Report:")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    return prob_metrics, hard