from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import numpy as np
from sklearn.model_selection import KFold


def get_metrics(model, X, y):
    kf = KFold(n_splits=5)
    acs, fs, ras = [], [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true = y_test
        acs.append(accuracy_score(y_true, y_pred))
        fs.append(f1_score(y_true, y_pred, average="micro"))
        ras.append(precision_score(y_true, y_pred, average="macro"))
    return np.mean(np.array(acs)), np.mean(np.array(fs)), np.mean(np.array(ras))
