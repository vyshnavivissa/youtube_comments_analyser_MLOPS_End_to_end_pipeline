from sentence_transformers import SentenceTransformer
import traceback

try:
    print("Starting...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )

    print("Model Loaded Successfully!")

except Exception as e:
    print("ERROR:")
    print(e)

    traceback.print_exc()

input("Press Enter to Exit...")