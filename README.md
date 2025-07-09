# SimPulse 🔍

**SimPulse** is a lightweight AI-powered semantic similarity engine designed to enhance claim verification in the **Smart Lost & Found System** (Available in my Library). It uses sentence-transformer embeddings and a machine learning regressor to compare user-submitted claims with found item descriptions and return a match score.

---

### 🚀 Key Features
- Trained on custom claim-description pairs (500+ entries)
- Embedding model: `all-MiniLM-L6-v2`
- Prediction model: `MLPRegressor`
- Predicts semantic match score between two text inputs
- Modular design for backend integration or API deployment

---

### 🧠 Purpose
This model was **specifically built** to improve claim ranking and decision-making in the **Smart Lost & Found System**, enabling faster and more intelligent resolution of lost item claims through AI.

---

### ⚙️ Usage
```python
from simPulse_code.predict import predict_similarity

score = predict_similarity(
    "black wallet with 3 cards and a zipped pocket",
    "I lost a small black wallet with some cards inside"
)

print(f"Predicted Match Score: {score}")
