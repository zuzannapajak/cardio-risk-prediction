from typing import Dict, Optional, Union
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 'pre' can be "passthrough" or a ColumnTransformer/Pipeline
def make_rf_fixed(pre: Union[str, Pipeline], seed: int = 42) -> Pipeline:
    """Fixed-config RandomForest pipeline (works for binary & multiclass)."""
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    return Pipeline([("pre", pre), ("clf", rf)])

def make_rf_base(pre: Union[str, Pipeline], seed: int = 42) -> Pipeline:
    """Base RF (untuned) for RandomizedSearchCV to modify via param grid."""
    rf = RandomForestClassifier(n_jobs=-1, random_state=seed)
    return Pipeline([("pre", pre), ("clf", rf)])

def rf_search_space() -> Dict[str, list]:
    """RandomizedSearchCV space (binary or multiclass)."""
    return {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2"],
        "clf__bootstrap": [True],
    }
