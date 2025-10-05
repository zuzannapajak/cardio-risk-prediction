from typing import Dict, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from .metrics import eval_probs

def fit_eval(
    model: Pipeline,
    X_tr,
    y_tr,
    X_te,
    y_te,
    sample_weight=None
) -> Dict[str, float]:
    """
    Fit a pipeline and return probability-based metrics on the test set.
    Works for binary and multiclass.
    """
    if sample_weight is not None:
        model.fit(X_tr, y_tr, **{"sample_weight": sample_weight})
    else:
        model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)

    classes: Optional[np.ndarray] = None
    try:
        if isinstance(model, Pipeline) and "clf" in model.named_steps:
            classes = model.named_steps["clf"].classes_
        else:
            classes = model.classes_
    except Exception:
        pass

    return eval_probs(y_te, proba, classes=classes)
