"""
trend_analysis.py
-----------------
Tracks discussion volume over time for key topics.
Detects sudden surges using z-score on weekly post counts.
Produces MAIN VISUALIZATION 4.

Spike detection method:
  - Aggregate UNIQUE posts per topic per week (deduplicated on Cleaned_Text)
  - Compute a 4-week rolling mean and std (shifted by 1 week to avoid lookahead)
  - Flag weeks where z-score > 2.0 as unusual surges
  - Report as "an unusual surge in discussion occurred" — no causal claims
    are made about why the surge happened unless the data itself shows it.

Trend chart:
  - Shows only the top 4 most-discussed topics for readability
  - Applies a 2-week rolling average to smooth week-to-week noise
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _weekly_counts_dedup(df_dedup: pd.DataFrame, term: str) -> pd.Series:
    """
    Returns a weekly time-series of UNIQUE post counts for a given topic term.
    Uses the deduplicated DataFrame so each unique tweet text counts once.
    """
    pattern = r"\b" + re.escape(term) + r"\b"
    mask    = df_dedup["Cleaned_Text"].str.contains(pattern, regex=True, na=False)
    subset  = df_dedup.loc[mask].copy()
    subset["Week"] = subset["Timestamp"].dt.to_period("W").dt.start_time
    return subset.groupby("Week").size().rename(term)


def detect_spikes(series: pd.Series, z_threshold: float = 2.0) -> pd.DataFrame:
    """
    Detects weeks where discussion volume is unusually high.

    Method: z-score relative to a 4-week rolling baseline (no lookahead).
    A week is flagged if its z-score exceeds z_threshold.

    Returns a DataFrame with columns:
      Week, Posts, Baseline, Z_Score, Pct_Increase
    Note: results are described as unusual surges, not attributed to causes.
    """
    if len(series) < 4:
        return pd.DataFrame()

    rolling_mean = series.rolling(window=4, min_periods=2).mean().shift(1)
    rolling_std  = series.rolling(window=4, min_periods=2).std().shift(1)

    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    spikes   = z_scores[z_scores > z_threshold].dropna()

    if spikes.empty:
        return pd.DataFrame()

    rows = []
    for week, z in spikes.items():
        baseline = rolling_mean.get(week, np.nan)
        count    = int(series.get(week, 0))
        pct_inc  = (
            (count - baseline) / baseline * 100
            if (baseline and not np.isnan(baseline) and baseline > 0)
            else None
        )
        rows.append({
            "Week"        : week,
            "Posts"       : count,
            "Baseline"    : round(float(baseline), 1) if (baseline and not np.isnan(baseline)) else None,
            "Z_Score"     : round(float(z), 2),
            "Pct_Increase": round(pct_inc, 1) if pct_inc is not None else None,
        })
    return pd.DataFrame(rows).sort_values("Z_Score", ascending=False)


def run_trend_analysis(df: pd.DataFrame,
                       top_topics: pd.Series,
                       plots_dir: str,
                       df_dedup: pd.DataFrame = None) -> dict:
    """
    Tracks top topics over time and detects unusual surges.
    Produces MAIN VIZ 4 and a diagnostic weekly volume plot.

    Parameters
    ----------
    df         : full cleaned DataFrame
    top_topics : Series of meaningful topics from topic_analysis
    plots_dir  : output folder
    df_dedup   : deduplicated DataFrame (one row per unique Cleaned_Text)
    """
    print("\n" + "=" * 60)
    print("STEP 6 & 7 - TOPIC SURGES & TRENDS OVER TIME")
    print("=" * 60)

    # Use deduplicated posts for trend counting
    source = df_dedup if df_dedup is not None else df
    if df_dedup is not None:
        print(f"\nUsing deduplicated posts: {len(source):,} unique tweet texts")
    else:
        print("\nWARNING: df_dedup not provided, using full dataset.")

    # Select top 4 topics for the trend chart (readable, not overcrowded)
    selected_topics = list(top_topics.head(4).index)
    print(f"\nTopics selected for trend chart: {selected_topics}")

    # Build weekly time-series for each topic
    weekly_series = {}
    for term in selected_topics:
        s = _weekly_counts_dedup(source, term)
        if len(s) >= 4:
            weekly_series[term] = s

    if not weekly_series:
        print("Not enough data for trend analysis.")
        return {}

    trend_df = pd.DataFrame(weekly_series).fillna(0).sort_index()

    # ── Surge detection ───────────────────────────────────────────────────────
    print("\nUNUSUAL SURGES IN DISCUSSION")
    print("(z-score > 2.0 above 4-week rolling baseline)")
    print("Note: surges are reported as observed volume anomalies.")
    print("No causal attribution is made from the data alone.")
    all_spikes = {}
    for term, series in weekly_series.items():
        spikes = detect_spikes(series, z_threshold=2.0)
        if not spikes.empty:
            all_spikes[term] = spikes
            print(f"\n  Topic: '{term}'")
            print(f"  An unusual surge in discussion occurred on the following weeks:")
            print(spikes.to_string(index=False))

    if not all_spikes:
        print("  No significant surges detected at z > 2.0 threshold.")

    # ------------------------------------------------------------------
    # MAIN VISUALIZATION 4: Topic Trends Over Time
    # ------------------------------------------------------------------
    # 2-week rolling average to smooth noise
    smooth_df = trend_df.rolling(window=2, min_periods=1).mean()
    colors    = ["#2980b9", "#e74c3c", "#27ae60", "#f39c12"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, topic in enumerate(smooth_df.columns):
        ax.plot(smooth_df.index, smooth_df[topic],
                label=topic, color=colors[i % len(colors)],
                linewidth=2.2, alpha=0.88)

    ax.set_title("MAIN VIZ 4 - Topic Trends Over Time (2-Week Rolling Average)\n"
                 "(unique tweet texts per week)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Number of Unique Posts", fontsize=11)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "viz4_topic_trends_over_time.png"), dpi=150)
    plt.close()
    print("\n[Saved] viz4_topic_trends_over_time.png")

    # ── Diagnostic: overall weekly volume (full dataset) ─────────────────────
    df_copy = df.copy()
    df_copy["Week"] = df_copy["Timestamp"].dt.to_period("W").dt.start_time
    weekly_total = df_copy.groupby("Week").size()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly_total.index, weekly_total.values,
            color="#2c3e50", linewidth=1.5)
    ax.fill_between(weekly_total.index, weekly_total.values,
                    alpha=0.15, color="#2c3e50")
    ax.set_title("Diagnostic - Overall Weekly Discussion Volume (all posts)", fontsize=12)
    ax.set_xlabel("Week")
    ax.set_ylabel("Posts")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "diagnostic_weekly_volume.png"), dpi=120)
    plt.close()
    print("[Saved] diagnostic_weekly_volume.png")

    return {
        "trend_df"  : trend_df,
        "all_spikes": all_spikes,
    }
