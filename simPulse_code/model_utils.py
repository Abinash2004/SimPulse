from sentence_transformers import SentenceTransformer
import numpy as np

# Load model only once (global)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    """Convert a text string into an embedding vector"""
    return model.encode(text, convert_to_numpy=True)

def build_features(text1: str, text2: str) -> np.ndarray:
    """Build feature vector for two texts: [emb1, emb2, abs(emb1 - emb2)]"""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    features = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)])
    return features
