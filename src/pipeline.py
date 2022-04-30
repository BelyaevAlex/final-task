from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def create_pipeline(log_reg: bool, use_scaler: bool, max_iter: int, penalty: str, n_neighbors: int, pca: bool) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        if pca:
            pipeline_steps.append(("scaler", PCA()))
        else:
            pipeline_steps.append(("scaler", StandardScaler()))
    if log_reg:
        pipeline_steps.append(
            ("classifier", LogisticRegression(max_iter=max_iter, penalty=penalty))
        )
    else:
        pipeline_steps.append(
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors))
        )
    return Pipeline(steps=pipeline_steps)