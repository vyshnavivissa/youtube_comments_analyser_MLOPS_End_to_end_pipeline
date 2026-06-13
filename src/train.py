import pandas as pd

df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None
)

df = df[[0,5]]

df.columns = [
    "sentiment",
    "text"
]

# Convert labels
df["sentiment"] = df["sentiment"].replace(
{
    0:0,
    4:1
}
)

print(df.head())

print("\nClass Distribution:")
print(df["sentiment"].value_counts())