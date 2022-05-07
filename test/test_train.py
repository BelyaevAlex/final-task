from click.testing import CliRunner
import pytest
import joblib
import sklearn
import pandas as pd
from src.train import train
from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    result = runner.invoke(
        train,
        [
            "--test_size",
            18,
            '--register_model',
            'False'
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '-t' / '--test_size'" in result.output
    if result.exit_code == 2:
        print('Test 1 success')

def test_all_good(
) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    result = runner.invoke(
        train,
        []
    )
    assert result.exit_code == 0

def test_save_model(
) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    path = 'model.joblib'
    result = runner.invoke(
        train,
        [
            "--save-model-path",
            path,
            '--register_model',
            'False'
        ]
    )
    model = joblib.load(path)
    assert str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>"
    if str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>":
        print('Test 2 success')

def test_input_data(
) -> None:
    data = [[13457, 2696, 110, 8, 60, 300, 60, 230, 234,
             138, 828, 0, 0, 1, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 5],
            [13458, 2335, 27, 29, 30, 12, 451, 194, 163,
             90, 666, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 6],
            [13459, 2437, 66, 25, 30, -5, 824, 235, 182,
             64, 859, 0, 0, 0, 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 3],
            [13460, 2416, 53, 24, 0, 0, 774, 226, 188,
             85, 886, 0, 0, 0, 1, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 4]]
    data = pd.DataFrame(data, columns=['Id', 'Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
       'Cover_Type'])
    data.to_csv('train.csv')
    runner = CliRunner()
    path = 'train.csv'
    result = runner.invoke(
        train,
        [
            '--dataset-path',
            path,
            '--register_model',
            'False'
        ],
    )
    assert result.exit_code == 0
 