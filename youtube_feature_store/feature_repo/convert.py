import pandas as pd

df = pd.read_csv("data/comments.csv")

df["event_timestamp"] = pd.to_datetime(
    df["event_timestamp"]
)

df.to_parquet(
    "data/comments.parquet",
    index=False
)

print("Parquet file created successfully!")