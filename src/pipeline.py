from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def create_pipeline(use_scaler: bool, max_iter: int, penalty: str) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        ("classifier", LogisticRegression(max_iter=max_iter, penalty=penalty))
    )
    return Pipeline(steps=pipeline_steps)