import numpy as np
import pandas as pd
from pathlib import Path
import click
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)


def train(
    dataset_path: Path
):
    data = pd.read_csv(dataset_path)
    print(data.columns)
    x = data.copy().drop(columns='Cover_Type')
    y = data['Cover_Type']
    model = LogisticRegression()
    model.fit(x, y)