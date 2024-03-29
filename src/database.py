import pandas as pd
from typing import Tuple
from pathlib import Path
import pandas_profiling

a = Tuple[pd.DataFrame, pd.Series]


def get_dataset(dataset_path: Path, prof: bool) -> a:
    data = pd.read_csv(dataset_path)
    x = data.copy().drop(columns="Cover_Type")
    y = data["Cover_Type"]
    if prof:
        pr = pandas_profiling.ProfileReport(data)
        pr.to_file(output_file="data.html")
    return x, y
