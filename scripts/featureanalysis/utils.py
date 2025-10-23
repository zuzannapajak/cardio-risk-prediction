import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42

def _safe_n_splits(y: pd.Series, n_splits: int) -> int:
    """Reduce n_splits if any class has too few samples."""
    min_count = int(y.value_counts().min())
    if min_count < n_splits:
        n_new = max(2, min_count)  # need at least 2 folds for StratifiedKFold
        print(f"Reducing n_splits from {n_splits} to {n_new} due to rare class counts.")
        return n_new
    return n_splits

def freeze_stratified_folds(
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = SEED,
    shuffle: bool = True,
):
    """Return (skf, folds) where folds is a list of (train_idx, val_idx) arrays."""
    n_splits_safe = _safe_n_splits(y, n_splits)
    skf = StratifiedKFold(n_splits=n_splits_safe, shuffle=shuffle, random_state=random_state)
    idx = np.arange(len(y))
    folds = [(tr, va) for tr, va in skf.split(idx, y)]
    return skf, folds

def _summarize_cv(name: str, y: pd.Series, folds):
    counts = y.value_counts().sort_index()
    print(f"\n[{name}] classes & counts:")
    print(counts.to_string())
    print(f"[{name}] n_folds: {len(folds)}")
    # Quick check that stratification held
    for i, (tr, va) in enumerate(folds, start=1):
        c_tr = y.iloc[tr].value_counts(normalize=True).sort_index()
        c_va = y.iloc[va].value_counts(normalize=True).sort_index()
        print(f"  • Fold {i}: train dist -> {c_tr.round(3).to_dict()} | val dist -> {c_va.round(3).to_dict()}")
