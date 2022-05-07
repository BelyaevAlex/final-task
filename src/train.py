from pathlib import Path
import click
import mlflow
import mlflow.sklearn
import numpy as np
from joblib import dump
from .pipeline import create_pipeline
from .database import get_dataset
from sklearn.model_selection import KFold
from .search_hyperparametrs import get_parameters
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="./data/train.csv",
    type=click.Path(path_type=Path),
)
@click.option("-p", "--penalty", default="l2", type=str)
@click.option("-m", "--max_iter", default=100, type=int)
@click.option(
    "-s",
    "--save-model-path",
    default="./data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "-u",
    "--use_scaler",
    default=True,
    type=bool,
)
@click.option(
    "-sd",
    "--split_dataset",
    default=False,
    type=bool,
)
@click.option(
    "-t",
    "--test_size",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
@click.option(
    "-lg",
    "--log_reg",
    default=True,
    type=bool,
)
@click.option(
    "-n",
    "--n_neighbors",
    default=5,
    type=int,
)
@click.option(
    "-pca",
    "--pca",
    default=False,
    type=bool,
)
@click.option(
    "-gs",
    "--gridsearch",
    default=False,
    type=bool,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: bool,
    test_size: float,
    split_dataset: bool,
    penalty: str,
    max_iter: int,
    log_reg: bool,
    n_neighbors: int,
    pca: bool,
    gridsearch: bool,
) -> None:
    with mlflow.start_run():
        pipeline = create_pipeline(
            log_reg, use_scaler, max_iter, penalty, n_neighbors, pca
        )
        x, y = get_dataset(dataset_path)
        pipeline.fit(x, y)
        dump(pipeline, save_model_path)
        print(f"Model was save in {save_model_path}")
        kf = KFold(n_splits=4)
        acs, fs, ras = [], [], []
        X = x
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_true = y_test
            acs.append(accuracy_score(y_true, y_pred))
            fs.append(f1_score(y_true, y_pred, average="micro"))
            ras.append(precision_score(y_true, y_pred, average="macro"))
        acs = np.mean(np.array(acs))
        fs = np.mean(np.array(fs))
        ras = np.mean(np.array(ras))
        print(f"accuracy is {acs}, f1 is {fs}, precision is {ras}")
        if gridsearch:
            print(
                get_parameters(log_reg=log_reg, x=pd.DataFrame(x), y=pd.Series(y))[0]
            )  # {'C': 5, 'penalty': 'l2', 'solver': 'newton-cg'},
            # KNN {'algorithm': 'auto', 'n_neighbors': 1, 'weights': 'uniform'}
        mlflow.log_param("PCA", pca)
        mlflow.log_param("use_scaler", use_scaler)
        if log_reg:
            mlflow.log_param("penalty", penalty)
            mlflow.log_param("max_iter", max_iter)
        else:
            mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("accuracy", acs)
        mlflow.log_metric("f1", fs)
        mlflow.log_metric("precision", ras)
