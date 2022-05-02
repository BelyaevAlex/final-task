from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Tuple
from typing import Any


def get_metrics(model: Any, x: Any, y: Any) -> Tuple[list[Any], list[Any], list[Any]]:
    kf = KFold(n_splits=5)
    acs, fs, ras = [], [], []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_true = y_test
        acs.append(accuracy_score(y_true, y_pred))
        fs.append(f1_score(y_true, y_pred, average="micro"))
        ras.append(precision_score(y_true, y_pred, average="macro"))
    return acs, fs, ras
