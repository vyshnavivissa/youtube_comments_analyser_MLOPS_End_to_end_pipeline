from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel

import joblib
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from youtube_comment_downloader import YoutubeCommentDownloader
from wordcloud import WordCloud

from prometheus_client import Counter, generate_latest


print("Loading Logistic Regression Model...")

model = joblib.load(
    "models/logistic_regression_model.pkl"
)

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Models Loaded Successfully!")

app = FastAPI(
    title="YouTube Comment Analyzer API",
    version="1.0"
)

# =========================
# Prometheus Metrics
# =========================

prediction_counter = Counter(
    "predictions_total",
    "Total predictions made"
)

analyze_counter = Counter(
    "analyze_requests_total",
    "Total analyze requests"
)

successful_requests = Counter(
    "successful_requests_total",
    "Total successful requests"
)

positive_predictions = Counter(
    "positive_predictions_total",
    "Total positive predictions"
)

negative_predictions = Counter(
    "negative_predictions_total",
    "Total negative predictions"
)

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Static Folder
# =========================

if not os.path.exists("static"):
    os.makedirs("static")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# =========================
# Request Models
# =========================

class CommentRequest(BaseModel):
    text: str


class YouTubeRequest(BaseModel):
    youtube_url: str


# =========================
# Routes
# =========================

@app.get("/")
def home():
    return {
        "message": "YouTube Comment Analyzer API Running"
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )


# =========================
# Single Comment Prediction
# =========================

@app.post("/predict")
def predict(request: CommentRequest):

    prediction_counter.inc()

    embedding = embedding_model.encode(
        [request.text]
    )

    prediction = model.predict(
        embedding
    )[0]

    successful_requests.inc()

    if prediction == 1:
        positive_predictions.inc()
    else:
        negative_predictions.inc()

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "comment": request.text,
        "sentiment": sentiment
    }


# =========================
# YouTube Comment Analysis
# =========================

@app.post("/analyze")
def analyze(request: YouTubeRequest):

    analyze_counter.inc()

    downloader = YoutubeCommentDownloader()

    comments = []

    for comment in downloader.get_comments_from_url(
        request.youtube_url
    ):
        comments.append(
            comment["text"]
        )

        if len(comments) >= 100:
            break

    if len(comments) == 0:
        return {
            "error": "No comments found"
        }

    embeddings = embedding_model.encode(
        comments,
        show_progress_bar=False
    )

    predictions = model.predict(
        embeddings
    )

    positive = int(
        np.sum(predictions == 1)
    )

    negative = int(
        np.sum(predictions == 0)
    )

    total = len(predictions)

    prediction_counter.inc(total)

    successful_requests.inc()

    positive_predictions.inc(
        positive
    )

    negative_predictions.inc(
        negative
    )

    # Generate Word Cloud
    text = " ".join(comments)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(text)

    wordcloud_path = "static/wordcloud.png"

    wordcloud.to_file(
        wordcloud_path
    )

    return {
        "youtube_url": request.youtube_url,
        "total_comments": total,
        "positive_comments": positive,
        "negative_comments": negative,
        "positive_percentage": round(
            (positive / total) * 100,
            2
        ),
        "negative_percentage": round(
            (negative / total) * 100,
            2
        ),
        "comments": comments,
        "wordcloud_url": "http://127.0.0.1:8000/static/wordcloud.png"
    }