import itertools, numpy as np
from typing import Dict, Mapping, Sequence
from sklearn.base import ClassifierMixin
from .evaluation import oof_scores, find_best_threshold


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
    # OOF scores for each base model
    oof = {name: oof_scores(mdl, X, y, cv) for name, mdl in models_dict.items()}
    names = list(models_dict.keys())

    best = {"weights": None, "threshold": 0.5, "metric": -1.0}

    for weights in itertools.product(weight_range, repeat=len(names)):
        w = np.asarray(weights, dtype=float)
        sumw = w.sum()
        if sumw <= 0:
            continue  # skip the all-zero vector

        # weighted average of OOF scores
        S = sum(w[i] * oof[names[i]] for i in range(len(names))) / sumw

        thr, val = find_best_threshold(y, S, metric=metric)
        if val > best["metric"]:
            best = {
                "weights": dict(zip(names, weights)),
                "threshold": thr,
                "metric": val,
            }

    return best
