"""
sentiment_analysis.py
---------------------
Topic Engagement Analysis.
For each meaningful topic, calculates engagement statistics from the
DEDUPLICATED post set (one row per unique tweet text).

Why deduplicated?
  A viral tweet retweeted 3,000 times has one engagement value repeated
  3,000 times. Using the full dataset would make median engagement for
  that topic equal to the viral tweet's score, which is misleading.
  Using deduplicated posts gives a truer picture of how engaging the
  TOPIC is across distinct authors.

Produces MAIN VISUALIZATION 3.
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_engagement_analysis(df: pd.DataFrame,
                             top_topics: pd.Series,
                             plots_dir: str,
                             df_dedup: pd.DataFrame = None) -> pd.DataFrame:
    """
    For each topic in top_topics, find all UNIQUE posts containing that term
    and compute engagement statistics independently.

    Parameters
    ----------
    df       : full cleaned DataFrame (used for context)
    top_topics : Series of topic -> frequency from topic_analysis
    plots_dir  : output folder for plots
    df_dedup   : deduplicated DataFrame (one row per unique Cleaned_Text).
                 If None, falls back to df (with a warning).

    Returns a DataFrame of topic engagement stats.
    """
    print("\n" + "=" * 60)
    print("STEP 5 - TOPIC ENGAGEMENT ANALYSIS")
    print("=" * 60)

    # Use deduplicated posts if available
    if df_dedup is not None:
        source = df_dedup
        print(f"\nUsing deduplicated posts: {len(source):,} unique tweet texts")
    else:
        source = df
        print("\nWARNING: df_dedup not provided, using full dataset.")

    records = []
    for term in top_topics.index:
        pattern = r"\b" + re.escape(term) + r"\b"
        mask    = source["Cleaned_Text"].str.contains(pattern, regex=True, na=False)
        subset  = source.loc[mask, "Engagement"]

        if len(subset) == 0:
            continue

        records.append({
            "Topic"            : term,
            "Post_Count"       : len(subset),
            "Total_Engagement" : int(subset.sum()),
            "Mean_Engagement"  : round(subset.mean(), 1),
            "Median_Engagement": subset.median(),
            "Min_Engagement"   : int(subset.min()),
            "Max_Engagement"   : int(subset.max()),
            "Unique_Eng_Values": subset.nunique(),
        })

    eng_df = pd.DataFrame(records).sort_values(
        "Median_Engagement", ascending=False
    ).reset_index(drop=True)

    # ── Validation table ──────────────────────────────────────────────────────
    # Printed before the chart to verify each topic has independently
    # calculated, distinct engagement statistics.
    print("\nEngagement Validation Table (sorted by Median):")
    print(f"{'Topic':<28} {'Posts':>6} {'Median':>8} {'Mean':>8} "
          f"{'Min':>6} {'Max':>8} {'UniqVals':>9}")
    print("-" * 80)
    for _, row in eng_df.iterrows():
        print(f"  {row['Topic']:<26} {row['Post_Count']:>6,} "
              f"{row['Median_Engagement']:>8.0f} {row['Mean_Engagement']:>8.0f} "
              f"{row['Min_Engagement']:>6,} {row['Max_Engagement']:>8,} "
              f"{row['Unique_Eng_Values']:>9,}")

    # Sanity check: flag any topics with suspiciously few unique engagement values
    suspicious = eng_df[eng_df["Unique_Eng_Values"] <= 3]
    if not suspicious.empty:
        print("\nNOTE: The following topics have very few unique engagement values,")
        print("which may indicate a viral retweet cluster rather than organic discussion:")
        for _, row in suspicious.iterrows():
            print(f"  '{row['Topic']}' — {row['Unique_Eng_Values']} unique values "
                  f"across {row['Post_Count']:,} posts")

    # ------------------------------------------------------------------
    # MAIN VISUALIZATION 3: Topic Engagement (Median vs Mean)
    # ------------------------------------------------------------------
    # Sort ascending so highest bar is at top of horizontal chart
    plot_df = eng_df.sort_values("Median_Engagement")

    fig, ax = plt.subplots(figsize=(10, 7))
    # Mean bar (lighter, behind)
    ax.barh(plot_df["Topic"], plot_df["Mean_Engagement"],
            color="#F0B27A", edgecolor="white", alpha=0.7, label="Mean")
    # Median bar (darker, in front)
    bars = ax.barh(plot_df["Topic"], plot_df["Median_Engagement"],
                   color="#E67E22", edgecolor="white", label="Median")

    for bar, val in zip(bars, plot_df["Median_Engagement"]):
        ax.text(bar.get_width() + max(plot_df["Median_Engagement"].max() * 0.01, 1),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)

    ax.set_title("MAIN VIZ 3 - Topic Engagement (Median vs Mean)\n"
                 "(based on unique tweet texts per topic)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Engagement Score", fontsize=11)
    ax.set_ylabel("Topic", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "viz3_topic_engagement.png"), dpi=150)
    plt.close()
    print("\n[Saved] viz3_topic_engagement.png")

    return eng_df
