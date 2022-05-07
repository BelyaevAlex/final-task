import pandas as pd
from typing import Tuple
from pathlib import Path


def get_dataset(dataset_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(dataset_path)
    x = data.copy().drop(columns="Cover_Type")
    y = data["Cover_Type"]
    return x, y
