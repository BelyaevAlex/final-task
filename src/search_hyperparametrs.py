from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import pandas as pd
from typing import Any


def get_parameters(log_reg: bool, x: Any, y: Any) -> list[Any]:
    cv = KFold(n_splits=3, shuffle=True, random_state=1)
    if log_reg:
        parameters = {
            "penalty": ("l2", "none"),
            "C": [1, 5, 10],
            "solver": ("newton-cg", "lbfgs", "liblinear", "sag", "saga"),
        }
        svc = LogisticRegression()
        clf = GridSearchCV(svc, parameters, cv=cv, scoring="f1_micro")
    else:
        parameters = {
            "n_neighbors": [1, 3, 5, 7, 9],
            "weights": ("uniform", "distance"),
            "algorithm": ("auto", "ball_tree", "kd_tree", "brute"),
        }
        svc = KNeighborsClassifier()
        clf = GridSearchCV(svc, parameters, cv=cv, scoring="f1_micro")
    clf.fit(x, y)
    return [clf.best_params_]
