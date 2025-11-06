import itertools, numpy as np, pandas as pd
from typing import Dict, Iterable, Mapping, Sequence, Tuple 
from sklearn.base import ClassifierMixin
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, hamming_loss, jaccard_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score, zero_one_loss)
from sklearn.model_selection import cross_val_predict
from IPython.display import display

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] with safe denominator."""
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    rng = xmax - xmin
    if not np.isfinite(rng) or rng <= 0:
        return np.zeros_like(x)
    return (x - xmin) / (rng + 1e-12)


def _scores_from_estimator(model: ClassifierMixin, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Return a probability-like score in [0,1]:
    - predict_proba[:,1] if available,
    - min-max normalized decision_function otherwise,
    - else numeric prediction cast to float.
    """
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        s = _safe_minmax(raw)
    else:
        s = model.predict(X).astype(float)
    return np.nan_to_num(s, nan=0.5, posinf=1.0, neginf=0.0)

# ---------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------
def oof_scores(model: ClassifierMixin, X, y, cv) -> np.ndarray:
    """
    Out-of-fold scores in [0, 1] (probabilities if available).
    """
    if hasattr(model, "predict_proba"):
        s = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = cross_val_predict(model, X, y, cv=cv, method="decision_function", n_jobs=-1)
        s = _safe_minmax(raw)
    else:
        s = cross_val_predict(model, X, y, cv=cv, method="predict", n_jobs=-1).astype(float)

    return np.nan_to_num(s, nan=0.5, posinf=1.0, neginf=0.0)


def evaluate_classification(
    model: ClassifierMixin,
    X_test: pd.DataFrame | np.ndarray,
    y_test: Iterable[int],
    threshold: float = 0.5,
    show: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute a comprehensive set of binary classification metrics and (optionally) display them.

    Returns
    -------
    df_metrics : pd.DataFrame
        Columns: ["Metric", "Value"]
    details : dict
        keys:
          - "cm" (pd.DataFrame 2x2 confusion matrix)
          - "y_pred" (np.ndarray of ints)
          - "y_score" (np.ndarray of floats in [0,1])
    """
    y_true = np.asarray(list(y_test), dtype=int)

    # probability-like scores
    y_score = _scores_from_estimator(model, X_test)
    y_pred = (y_score >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    # clip for log loss
    y_score_clipped = np.clip(y_score, 1e-15, 1 - 1e-15)
    probas = np.column_stack([1 - y_score_clipped, y_score_clipped])

    # metrics dict
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (pos=1)": precision_score(y_true, y_pred, zero_division=0),
        "Recall / Sensitivity (TPR)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity (TNR)": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC (Average Precision)": average_precision_score(y_true, y_score),
        "Log loss": log_loss(y_true, probas, labels=[0, 1]),
        "Jaccard": jaccard_score(y_true, y_pred, zero_division=0),
        "Hamming loss": hamming_loss(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's kappa": cohen_kappa_score(y_true, y_pred),
        "Zero-One loss": zero_one_loss(y_true, y_pred),
        "Brier score": brier_score_loss(y_true, y_score),
        "Threshold": float(threshold),
        "Positives (TP+FN)": int(tp + fn),
        "Negatives (TN+FP)": int(tn + fp),
    }

    df_metrics = pd.DataFrame({"Metric": list(metrics.keys()), "Value": np.round(list(metrics.values()), 4)})
    cm_df = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])

    if show:
        display(df_metrics)
        print("\nConfusion matrix:")
        display(cm_df)

    return df_metrics, {"cm": cm_df, "y_pred": y_pred, "y_score": y_score}


def find_best_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Grid search the best threshold for a chosen metric.

    Parameters
    ----------
    metric : {"f1", "mcc", "balanced_accuracy"}

    Returns
    -------
    best_threshold, best_metric
    """
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    
    metric = (metric or "f1").lower().strip()
    eps = 1e-12
    _, _, thr_pr = precision_recall_curve(y_true, y_score)
    candidates = np.unique(np.r_[np.min(y_score) - eps, thr_pr, np.max(y_score) + eps])

    def score_at(t: float) -> float:
        y_pred = (y_score >= t).astype(int)
        if metric == "mcc":
            return matthews_corrcoef(y_true, y_pred)
        if metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, y_pred)
        return f1_score(y_true, y_pred, zero_division=0)    # default

    scores = np.array([score_at(t) for t in candidates])
    i = int(np.nanargmax(scores))
    return float(candidates[i]), float(scores[i])


def best_soft_voting_setup(
    models_dict: Mapping[str, ClassifierMixin],
    X,
    y,
    cv,
    weight_range: Sequence[int] = range(0, 6),
    metric: str = "f1",
) -> Dict[str, object]:
    """
    Brute-force integer-weight search for a soft-voting ensemble on OOF scores,
    including per-weight optimal threshold tuning.

    Returns
    -------
    {
      "weights": {name: int_weight, ...},
      "threshold": float,
      "metric": float
    }
    """
    oof = {name: oof_scores(mdl, X, y, cv) for name, mdl in models_dict.items()}
    names = list(models_dict.keys())

    best = {"weights": None, "threshold": 0.5, "metric": -1.0}

    for weights in itertools.product(weight_range, repeat=len(names)):
        w = np.asarray(weights, dtype=float)
        sumw = w.sum()
        if sumw <= 0: continue  # skip the all-zero vector

        # weighted average of OOF scores
        S = sum(w[i] * oof[names[i]] for i in range(len(names))) / sumw

        thr, val = find_best_threshold(y, S, metric=metric)
        if val > best["metric"]:
            best = {"weights": dict(zip(names, weights)), "threshold": thr, "metric": val}

    return best
