from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def _scale_pos_weight(y: np.ndarray) -> float:
    # for binary only
    pos = int(np.sum(y))
    neg = int(len(y) - pos)
    return neg / max(pos, 1)


def make_xgb_pipeline(
    pre: Union[str, Pipeline],
    y_train: np.ndarray,
    seed: int = 42,
    task: str = "binary",              # "binary" | "multiclass"
    num_class: Optional[int] = None,   # required when task="multiclass"
) -> Pipeline:
    """
    Build an XGBClassifier pipeline with sensible defaults.
    - For binary: objective='binary:logistic', eval_metric='aucpr' (good with imbalance).
    - For multiclass: objective='multi:softprob', eval_metric='mlogloss'.
    """

    params = dict(
        tree_method="hist",
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=seed,
    )

    if task == "binary":
        params.update(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=_scale_pos_weight(y_train),
        )
    elif task == "multiclass":
        if num_class is None:
            num_class = int(len(np.unique(y_train)))
        params.update(
            objective="multi:softprob",
            num_class=num_class,
            eval_metric="mlogloss",
        )
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")

    clf = XGBClassifier(**params)
    return Pipeline([("pre", pre), ("clf", clf)])


def xgb_search_space(task: str = "binary") -> Dict[str, List]:
    """Common search space for both binary & multiclass."""
    space = {
        "clf__learning_rate": [0.02, 0.03, 0.05],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__min_child_weight": [1, 2, 3],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0],
        "clf__reg_alpha": [0.0, 0.25, 0.5, 1.0],
        "clf__n_estimators": [600, 900, 1200, 1500],  # early stopping will cap this
    }
    # For binary only, optionally explore scale_pos_weight
    if task == "binary":
        space["clf__scale_pos_weight"] = [0.5, 1.0, 2.0, 3.0]
    return space
