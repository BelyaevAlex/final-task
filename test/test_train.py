from click.testing import CliRunner
import pytest
import joblib
import sklearn
import os
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
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '-t' / '--test_size'" in result.output

def test_all_good(
) -> None:
    """It fails when test split ratio is greater than 1."""
    runner = CliRunner()
    result = runner.invoke(
        train,
        [],
    )
    assert result.exit_code == 0

def test_save_model(
) -> None:
    """It fails when test split ratio is greater than 1."""
    path_cd = os.getcwd()
    os.makedirs("test_model/")
    runner = CliRunner()
    print(Path(path_cd + '/test_model/'))
    path = Path(path_cd + '/test_model/model.joblib')
    result = runner.invoke(
        train,
        [
            "--save-model-path",
            path
        ]
    )
    model = joblib.load(path)
    os.remove(path)
    os.rmdir(path_cd + '/test_model/')
    assert str(type(model)) == "<class 'sklearn.pipeline.Pipeline'>"
