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
def train(
    dataset_path: Path,
    save_model_path: Path,
    use_scaler: bool,
    test_size: float,
    split_dataset: bool,
    penalty: str,
    max_iter: int
):
    pipeline = create_pipeline(use_scaler, max_iter, penalty)
    if split_dataset:
        x, x_test, y, y_test = get_dataset(dataset_path, split_dataset, test_size)
    else:
        x, y = get_dataset(dataset_path, split_dataset, test_size)
    pipeline.fit(x, y)
    dump(pipeline, save_model_path)
    print(f'Model was save in {save_model_path}')
    acs, fs, ras = get_metrics(pipeline, x, y)
    print(f'accuracy is {acs}, f1 is {fs}, precision is {ras}')