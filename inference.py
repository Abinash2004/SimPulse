from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPRegressor
import joblib
import numpy as np

# Load embedding model + regressor
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
reg = joblib.load("model/simpulse_model.pkl")

def predict_similarity(text1, text2):
    emb1 = embedding_model.encode(text1)
    emb2 = embedding_model.encode(text2)
    features = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)]).reshape(1, -1)
    score = reg.predict(features)[0]
    return round(score * 100, 2)

# HF API interface
def inference(inputs):
    text1 = inputs.get("text1", "")
    text2 = inputs.get("text2", "")
    return {"score": predict_similarity(text1, text2)}
