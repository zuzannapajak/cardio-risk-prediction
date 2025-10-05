from typing import Dict, Optional
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

def get_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

# def randomized_search(
#     model: Pipeline,
#     param_space: Dict,
#     X,
#     y,
#     cv: Optional[StratifiedKFold] = None,
#     n_iter: int = 24,
#     seed: int = 42,
#     scoring: str = "average_precision",
# ):
#     """Run a RandomizedSearchCV on a pipeline."""
#     if cv is None:
#         cv = get_cv(seed=seed)
#     search = RandomizedSearchCV(
#         model, param_space, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=seed
#     )
#     search.fit(X, y)
#     return search

def randomized_search(
    model: Pipeline,
    param_space: Dict,
    X,
    y,
    cv: Optional[StratifiedKFold] = None,
    n_iter: int = 24,
    seed: int = 42,
    scoring: Optional[str] = None,
    fit_params: Optional[Dict] = None,       # ← NEW
):
    """Run a RandomizedSearchCV on a pipeline."""
    if cv is None:
        cv = get_cv(seed=seed)
    search = RandomizedSearchCV(
        model, param_space, n_iter=n_iter, cv=cv,
        scoring=scoring, n_jobs=-1, random_state=seed,
        refit=True, return_train_score=False
    )
    if fit_params is None:
        fit_params = {}
    search.fit(X, y, **fit_params)            # ← pass early-stopping params here
    return search