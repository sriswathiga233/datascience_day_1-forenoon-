"""
data_cleaning.py
----------------
Loads the raw CSV, validates it, cleans the Text column,
and saves a cleaned copy. The original file is never modified.
"""

import re
import pandas as pd


# Twitter-specific noise words to exclude from analysis
TWITTER_NOISE = {
    "rt", "amp", "via", "http", "https", "co", "www",
    "twitter", "tweet", "retweet", "follow", "like", "share",
    "gt", "lt", "quot",
}


def clean_text(text: str) -> str:
    """
    Clean a single tweet string.
    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove RT prefix
      4. Remove @mentions
      5. Remove punctuation (keep letters, digits, spaces, #)
      6. Collapse extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)      # remove URLs
    text = re.sub(r"^rt\s+", " ", text)               # remove leading RT
    text = re.sub(r"@\w+", " ", text)                 # remove @mentions
    text = re.sub(r"[^a-z0-9\s#]", " ", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text


def load_and_clean(raw_path: str, cleaned_path: str) -> pd.DataFrame:
    """
    Full cleaning pipeline.
    Returns the cleaned DataFrame and prints a validation report.
    """
    print("=" * 60)
    print("STEP 1 - DATA CLEANING & VALIDATION")
    print("=" * 60)

    # Load
    df_raw = pd.read_csv(raw_path)
    print(f"\nRaw dataset loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    # Validation checks
    print("\n--- Validation Report ---")

    missing = df_raw.isnull().sum()
    print(f"Missing values:\n{missing}")

    dup_rows = df_raw.duplicated().sum()
    print(f"\nDuplicate rows: {dup_rows}")

    dup_ids = df_raw["Post_ID"].duplicated().sum()
    print(f"Duplicate Post_IDs: {dup_ids}")

    invalid_eng = (df_raw["Engagement"] < 0).sum()
    print(f"Invalid Engagement values (< 0): {invalid_eng}")

    # Work on a copy - never touch the original
    df = df_raw.copy()

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    ts_nulls = df["Timestamp"].isnull().sum()
    if ts_nulls:
        print(f"\nTimestamp parse failures (dropped): {ts_nulls}")
        df = df.dropna(subset=["Timestamp"])

    # Drop negative engagement rows if any
    if invalid_eng > 0:
        df = df[df["Engagement"] >= 0]

    # Clean Text
    print("\nCleaning Text column ...")
    df["Cleaned_Text"] = df["Text"].apply(clean_text)

    # Drop rows where cleaning left an empty string
    before = len(df)
    df = df[df["Cleaned_Text"].str.strip() != ""]
    after = len(df)
    print(f"Rows removed (empty after cleaning): {before - after:,}")

    # Summary
    print(f"\nBefore cleaning : {df_raw.shape[0]:,} rows")
    print(f"After  cleaning : {after:,} rows")
    print(f"Rows removed    : {df_raw.shape[0] - after:,}")

    # Save cleaned dataset
    df.to_csv(cleaned_path, index=False)
    print(f"\nCleaned dataset saved -> {cleaned_path}")
    print("=" * 60)

    return df
