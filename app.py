import gradio as gr
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
reg = joblib.load("model/simpulse_model.pkl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def predict(text1, text2):
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    features = np.concatenate([emb1, emb2, np.abs(emb1 - emb2)]).reshape(1, -1)
    score = reg.predict(features)[0]
    return round(score * 100, 2)

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=["text", "text"],
    outputs="number",
    title="SimPulse: Semantic Sentence Matcher",
    description="Returns a semantic similarity score between a two sentencesclear. Built for AI integration in Smart Lost & Found System."
)

iface.launch()