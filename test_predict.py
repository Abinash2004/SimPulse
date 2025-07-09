from simPulse_code.predict import predict_similarity

score = predict_similarity(
    "black wallet with 3 cards and a zipped pocket",
    "I lost a black pouch with cards"
)

print(f"Predicted Match Score: {score}")
