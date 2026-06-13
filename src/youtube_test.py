from youtube_comment_downloader import YoutubeCommentDownloader
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

print("Loading Model...")

xgb_model = joblib.load(
    "models/xgb_model.pkl"
)

embedding_model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2"
)

url = input("Enter YouTube URL: ")

print("Fetching Comments...")

downloader = YoutubeCommentDownloader()

comments = []

for comment in downloader.get_comments_from_url(url):
    comments.append(comment["text"])

    if len(comments) >= 100:
        break

print(f"Comments Retrieved: {len(comments)}")

print("Generating Embeddings...")

embeddings = embedding_model.encode(
    comments,
    show_progress_bar=True
)

predictions = xgb_model.predict(
    embeddings
)

positive = np.sum(predictions == 1)
negative = np.sum(predictions == 0)

print("\nRESULTS")
print("-" * 30)

print(
    f"Positive Comments: {positive}"
)

print(
    f"Negative Comments: {negative}"
)

print(
    f"Positive %: {(positive/len(predictions))*100:.2f}"
)

print(
    f"Negative %: {(negative/len(predictions))*100:.2f}"
)