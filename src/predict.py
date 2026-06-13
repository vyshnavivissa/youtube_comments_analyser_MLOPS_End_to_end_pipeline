import joblib
from sentence_transformers import SentenceTransformer

print("Loading XGBoost model...")

model = joblib.load(
    "models/xgb_model.pkl"
)

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Embedding model loaded.")

while True:

    comment = input("\nEnter Comment (or type exit): ")

    if comment.lower() == "exit":
        print("Exiting...")
        break

    embedding = embedding_model.encode(
        [comment]
    )

    prediction = model.predict(
        embedding
    )[0]

    if prediction == 1:
        print("Prediction: Positive ")

    else:
        print("Prediction: Negative ")