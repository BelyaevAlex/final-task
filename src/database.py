import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(dataset_path, split_dataset, test_size):
    data = pd.read_csv(dataset_path)
    x = data.copy().drop(columns="Cover_Type")
    y = data["Cover_Type"]
    if split_dataset:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        return x_train, x_test, y_train, y_test
    else:
        return x, y
