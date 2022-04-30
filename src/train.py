from pathlib import Path
import click
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .pipeline import create_pipeline
from .database import get_dataset
from .test_metrics import get_metrics
from .search_hyperparametrs import get_parameters

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)

@click.option(
    "-p",
    "--penalty",
    default="l2",
    type=str
)

@click.option(
    "-m",
    "--max_iter",
    default=100,
    type=int
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
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
    type=float,
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
    gridsearch: bool
):
    with mlflow.start_run():
        pipeline = create_pipeline(log_reg, use_scaler, max_iter, penalty, n_neighbors, pca)
        if split_dataset:
            x, x_test, y, y_test = get_dataset(dataset_path, split_dataset, test_size)
        else:
            x, y = get_dataset(dataset_path, split_dataset, test_size)
        pipeline.fit(x, y)
        dump(pipeline, save_model_path)
        print(f'Model was save in {save_model_path}')
        acs, fs, ras = get_metrics(pipeline, x, y)
        print(f'accuracy is {acs}, f1 is {fs}, precision is {ras}')
        if gridsearch:
            print(get_parameters(log_reg=log_reg, x=x, y=y).best_params_) #{'C': 5, 'penalty': 'l2', 'solver': 'newton-cg'}, KNN {'algorithm': 'auto', 'n_neighbors': 1, 'weights': 'uniform'}
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
        mlflow.sklearn.log_model(pipeline['classifier'], "model")
