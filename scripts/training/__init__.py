"""
Training utilities package
--------------------------

Provides:
- Cross-validation and randomized search tools
- Unified evaluation and reporting helpers
- Probability and hard-pred metrics for binary & multiclass tasks
"""

from .cv_search import get_cv, randomized_search
from .metrics import eval_probs
from .evaluation import fit_eval
from .reporting import print_eval_report

__all__ = [
    "get_cv",
    "randomized_search",
    "eval_probs",
    "fit_eval",
    "print_eval_report",
]
