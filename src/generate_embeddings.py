import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading Dataset...")

df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

df = df[[0, 5]]

df.columns = [
    "sentiment",
    "text"
]

df["sentiment"] = df["sentiment"].replace({
    0: 0,
    4: 1
})

# Take only 50k rows
df = df.sample(
    n=50000,
    random_state=42
)

print("Loading Embedding Model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating Embeddings...")

embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

print("Embedding Shape:")
print(embeddings.shape)