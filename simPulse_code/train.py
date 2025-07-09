import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib
import os
from simPulse_code.model_utils import get_embedding

def train_model():
    data_path = os.path.join("data", "training_data.csv")
    df = pd.read_csv(data_path)

    # Rename columns
    df = df.rename(columns={"sentence1": "desc1", "sentence2": "desc2", "matching": "score"})

    # Drop any rows with missing or non-string descriptions
    df = df.dropna(subset=["desc1", "desc2", "score"])
    df["desc1"] = df["desc1"].astype(str).str.strip()
    df["desc2"] = df["desc2"].astype(str).str.strip()

    if len(df) < 5:
        raise ValueError("❌ Not enough data to train. Please provide at least 5 rows.")

    emb1 = np.array([get_embedding(x) for x in df["desc1"]])
    emb2 = np.array([get_embedding(x) for x in df["desc2"]])

    X = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)], axis=1)
    y = df["score"].astype(float).values / 100

    reg = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )

    reg.fit(X, y)

    os.makedirs("model", exist_ok=True)
    joblib.dump(reg, "model/simpulse_model.pkl")
    print("✅ Model trained and saved at model/simpulse_model.pkl")
