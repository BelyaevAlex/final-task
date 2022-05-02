import joblib
import sklearn
model = joblib.load('../data/model.joblib')
print(type(model))