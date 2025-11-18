"""
Model training utilities:
- OOF scoring
- Comprehensive binary classification metrics
- Threshold optimization
- Soft-voting ensemble weight search
"""

from .evaluation import (
    oof_scores,
    evaluate_classification,
    find_best_threshold,
)
from .ensembles import best_soft_voting_setup

__all__ = [
    "oof_scores",
    "evaluate_classification",
    "find_best_threshold",
    "best_soft_voting_setup",
]
