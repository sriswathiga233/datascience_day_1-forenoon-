"""
eda.py
------
Exploratory Data Analysis.
Calculates and prints key statistics, and saves supporting plots.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def run_eda(df: pd.DataFrame, plots_dir: str) -> dict:
    """
    Runs EDA on the cleaned DataFrame.
    Returns a dict of key stats for use in insights / action plan.
    """
    print("\n" + "=" * 60)
    print("STEP 2 - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    os.makedirs(plots_dir, exist_ok=True)

    # Basic counts
    n_posts = len(df)
    date_min = df["Timestamp"].min().strftime("%Y-%m-%d")
    date_max = df["Timestamp"].max().strftime("%Y-%m-%d")

    print(f"\nTotal posts   : {n_posts:,}")
    print(f"Date range    : {date_min}  to  {date_max}")

    # Sentiment distribution
    sent_counts = df["Sentiment_Label"].value_counts()
    sent_pct    = (sent_counts / n_posts * 100).round(1)
    print("\nSentiment Distribution:")
    for label in sent_counts.index:
        print(f"  {label:<10} {sent_counts[label]:>7,}  ({sent_pct[label]}%)")

    # Engagement statistics
    eng = df["Engagement"]
    eng_stats = {
        "mean"  : round(eng.mean(), 2),
        "median": eng.median(),
        "min"   : eng.min(),
        "max"   : eng.max(),
    }
    print("\nEngagement Statistics:")
    for k, v in eng_stats.items():
        print(f"  {k:<8}: {v:,.2f}" if isinstance(v, float) else f"  {k:<8}: {v:,}")

    # Posts by month
    df["YearMonth"] = df["Timestamp"].dt.to_period("M")
    posts_by_month  = df.groupby("YearMonth").size().sort_index()
    print("\nPosts by Month (top 5 busiest):")
    print(posts_by_month.sort_values(ascending=False).head(5).to_string())

    # Engagement by sentiment
    eng_by_sent = df.groupby("Sentiment_Label")["Engagement"].agg(
        Total="sum", Mean="mean", Median="median"
    ).round(2)
    print("\nEngagement by Sentiment:")
    print(eng_by_sent.to_string())

    # Diagnostic plot: posts by month (not one of the 4 main plots)
    fig, ax = plt.subplots(figsize=(12, 4))
    posts_by_month.plot(ax=ax, color="#4C72B0", linewidth=2)
    ax.set_title("Posts per Month (Diagnostic)", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Posts")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "diagnostic_posts_by_month.png"), dpi=120)
    plt.close()

    stats = {
        "n_posts"       : n_posts,
        "date_min"      : date_min,
        "date_max"      : date_max,
        "sent_counts"   : sent_counts,
        "sent_pct"      : sent_pct,
        "eng_stats"     : eng_stats,
        "eng_by_sent"   : eng_by_sent,
        "posts_by_month": posts_by_month,
    }
    return stats
