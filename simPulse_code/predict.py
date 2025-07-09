import joblib
import os
from simPulse_code.model_utils import build_features

regressor = joblib.load(os.path.join("model", "simpulse_model.pkl"))

def predict_similarity(desc1, desc2):
    features = build_features(desc1, desc2)
    score = regressor.predict([features])[0]
    return round(score * 100, 2)
