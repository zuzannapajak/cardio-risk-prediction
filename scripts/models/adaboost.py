from typing import Dict, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Dict

def make_ada_pipeline(pre: ColumnTransformer, seed: int = 42) -> Pipeline:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
        n_estimators=300,
        learning_rate=0.05,
        random_state=seed
    )
    return Pipeline([("pre", pre), ("clf", ada)])

def ada_search_space() -> Dict[str, List]:
    return {
        "clf__n_estimators": [200, 300, 500],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__estimator__max_depth": [1, 2, 3],
    }
