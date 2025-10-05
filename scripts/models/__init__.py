"""
Model definitions and parameter spaces
--------------------------------------

This package provides model builder functions and parameter search spaces
for various algorithms. Each model has its own module (e.g., random_forest.py,
xgboost.py, adaboost.py) to keep the code modular and organized.
"""

from .random_forest import (
    make_rf_fixed,
    make_rf_base,
    rf_search_space
)

from .xgboost import (
    make_xgb_pipeline, 
    xgb_search_space
)

from .adaboost import (
    make_ada_pipeline,
    ada_search_space
)

__all__ = [
    # Random Forest
    "make_rf_fixed",
    "make_rf_base",
    "rf_search_space",
    
    # XgBoost
    "make_xgb_pipeline", 
    "xgb_search_space"
    
    # AdaBoost
    "make_ada_pipeline",
    "ada_search_space"
]
