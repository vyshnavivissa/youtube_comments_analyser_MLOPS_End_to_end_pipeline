import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression

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

# Sample 50k rows
df = df.sample(
    n=50000,
    random_state=42
)

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating Embeddings...")

X = embedding_model.encode(
    df["text"].tolist(),
    show_progress_bar=True
)

y = df["sentiment"]

print("Train/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

mlflow.set_experiment(
    "youtube-comment-sentiment"
)

with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "feature_type",
        "SentenceTransformerEmbeddings"
    )

    print("Training Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Predicting...")

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        "logistic_regression_model"
    )

    joblib.dump(
        model,
        "models/logistic_regression_model.pkl"
    )

    print("Model Saved!")

print("MLflow Run Completed!")