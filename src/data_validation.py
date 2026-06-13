import pandas as pd

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

print("Running Data Quality Checks...")

# Check 1: No nulls in sentiment
assert df["sentiment"].isnull().sum() == 0

# Check 2: No nulls in text
assert df["text"].isnull().sum() == 0

# Check 3: Sentiment values must be 0 or 4
assert set(df["sentiment"].unique()) == {0, 4}

# Check 4: Dataset should not be empty
assert len(df) > 0

print("All Validations Passed")
print(f"Rows: {len(df)}")