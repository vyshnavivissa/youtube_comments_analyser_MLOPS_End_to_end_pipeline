from sentence_transformers import SentenceTransformer
import time

print("================================")
print("Testing Embedding Model")
print("================================")

start_time = time.time()

print("Loading Model...")

model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2",
    device="cpu"
)

print("Model Loaded Successfully!")

texts = [
    "This video is amazing",
    "Worst tutorial ever"
]

print("Generating Embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings Generated Successfully!")

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nFirst Embedding:")
print(embeddings[0][:10])

print(
    f"\nTotal Time: {time.time() - start_time:.2f} seconds"
)