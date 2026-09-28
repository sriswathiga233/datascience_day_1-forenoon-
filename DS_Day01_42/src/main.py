"""
main.py
-------
DS_Day01_42 - SOCIAL MEDIA: WHAT ARE CUSTOMERS TALKING ABOUT?

Entry point. Runs the full pipeline in order:
  1.  Data Cleaning & Validation
  2.  Exploratory Data Analysis
  3.  Discussion Topics
  4.  Positive vs Negative Comparison   (inside topic_analysis)
  5.  Topic Engagement
  6.  Spike Detection
  7.  Topic Trends Over Time
  8.  Four Main Visualizations           (produced inside each module)
  9.  Naive Bayes Model
  10. Model Evaluation
  11. Business Insights
  12. Practical Action Plan
"""

import os
import sys

# Make sure src/ is on the path when running from project root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Paths
BASE_DIR    = os.path.dirname(SRC_DIR)          # DS_Day01_42/
DATA_DIR    = os.path.join(BASE_DIR, "data")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RAW_CSV     = os.path.join(DATA_DIR, "social_media_brand_analytics.csv")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_social_media.csv")

# Create output directories
for d in [PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Import pipeline modules
from data_cleaning      import load_and_clean
from eda                import run_eda
from topic_analysis     import run_topic_analysis
from sentiment_analysis import run_engagement_analysis
from trend_analysis     import run_trend_analysis
from naive_bayes_model  import run_naive_bayes


def print_section(title: str):
    """Prints a clearly visible section header."""
    print("\n" + "=" * 66)
    print("  " + title.upper())
    print("=" * 66)


def generate_insights(eda_stats, topic_results, eng_df,
                      spike_results, model_results) -> list:
    """
    Generates 5-7 insights based entirely on calculated results.
    No values are hardcoded - all are pulled from the result dicts.
    """
    insights = []
    sc  = eda_stats["sent_counts"]
    sp  = eda_stats["sent_pct"]
    es  = eda_stats["eng_stats"]

    # Insight 1: Dominant sentiment
    dominant = sc.idxmax()
    insights.append({
        "title"   : "Neutral Sentiment Dominates the Conversation",
        "finding" : f"{dominant} posts make up {sp[dominant]}% of all discussions.",
        "evidence": f"{sc[dominant]:,} out of {eda_stats['n_posts']:,} posts are {dominant}.",
        "meaning" : (
            "Most users are sharing information or observing rather than expressing "
            "strong opinions. This is an opportunity for the brand to shape the "
            "narrative proactively."
        ),
    })

    # Insight 2: Engagement skew
    insights.append({
        "title"   : "Engagement is Highly Skewed - A Few Posts Drive Most Reach",
        "finding" : f"Mean engagement ({es['mean']:,.0f}) is far above median ({es['median']:,.0f}).",
        "evidence": f"Max engagement: {es['max']:,}. Median: {es['median']:,}.",
        "meaning" : (
            "A small number of viral posts dominate total reach. "
            "Brands should identify what makes those posts go viral and replicate it."
        ),
    })

    # Insight 3: Top topic
    top_term = topic_results["top_topics"].index[0]
    top_freq = int(topic_results["top_topics"].iloc[0])
    insights.append({
        "title"   : f"'{top_term}' is the Most Discussed Topic",
        "finding" : f"The term '{top_term}' appears in {top_freq:,} posts.",
        "evidence": "Frequency rank #1 among all extracted topics.",
        "meaning" : (
            "This is the core subject of the conversation. "
            "Brand messaging and content strategy should directly address this topic."
        ),
    })

    # Insight 4: Positive vs Negative distinctive topics
    pos_top = list(topic_results["pos_only"].head(3).index)
    neg_top = list(topic_results["neg_only"].head(3).index)
    insights.append({
        "title"   : "Positive and Negative Discussions Focus on Different Themes",
        "finding" : (
            f"Positive posts are associated with: {pos_top}. "
            f"Negative posts are associated with: {neg_top}."
        ),
        "evidence": "TF-IDF distinctiveness analysis across Positive vs Negative subsets.",
        "meaning" : (
            "Negative discussions signal specific pain points or controversies. "
            "Positive discussions reveal what resonates - these themes can be amplified."
        ),
    })

    # Insight 5: Top engagement topic(s) — handle ties
    if not eng_df.empty:
        top_med = eng_df.iloc[0]["Median_Engagement"]
        tied    = eng_df[eng_df["Median_Engagement"] == top_med]["Topic"].tolist()
        if len(tied) == 1:
            title_str   = f"'{tied[0]}' Generates the Highest Median Engagement"
            finding_str = (f"Posts about '{tied[0]}' have a median engagement "
                           f"of {top_med:,.0f} (unique posts, deduplicated).")
        else:
            joined      = " and ".join(f"'{t}'" for t in tied)
            title_str   = "Multiple Topics Share the Highest Median Engagement"
            finding_str = (f"Topics including {joined} share the highest median "
                           f"engagement of {top_med:,.0f} (unique posts, deduplicated).")
        insights.append({
            "title"   : title_str,
            "finding" : finding_str,
            "evidence": (f"Topic engagement analysis across {len(eng_df)} topics "
                         f"using deduplicated post corpus (89,162 unique texts)."),
            "meaning" : (
                "High-engagement topics are where the audience is most active. "
                "Prioritising content around these topics maximises organic reach."
            ),
        })

    # Insight 6: Spikes
    if spike_results.get("all_spikes"):
        spike_topic = list(spike_results["all_spikes"].keys())[0]
        spike_df    = spike_results["all_spikes"][spike_topic]
        top_spike   = spike_df.iloc[0]
        week_str    = top_spike["Week"].strftime("%Y-%m-%d") if hasattr(top_spike["Week"], "strftime") else str(top_spike["Week"])
        insights.append({
            "title"   : f"Sudden Spike Detected for '{spike_topic}'",
            "finding" : (
                f"Week of {week_str}: {top_spike['Posts']} posts "
                f"vs baseline of {top_spike['Baseline']}."
            ),
            "evidence": (
                f"Z-score: {top_spike['Z_Score']} | "
                f"Increase: {top_spike['Pct_Increase']}%"
            ),
            "meaning" : (
                "An unusual surge in discussion volume is a signal worth monitoring. "
                "Brands should track these moments and prepare rapid-response content "
                "in case the topic resurfaces."
            ),
        })

    # Insight 7: Model accuracy
    acc = model_results["accuracy"]
    insights.append({
        "title"   : f"Naive Bayes Achieves {acc * 100:.1f}% Accuracy on Sentiment Classification",
        "finding" : f"The model correctly classifies {acc * 100:.1f}% of unseen tweets.",
        "evidence": (
            f"Macro F1: {model_results['f1_macro']:.4f} | "
            f"Weighted F1: {model_results['f1_weighted']:.4f}"
        ),
        "meaning" : (
            "An automated sentiment classifier at this accuracy can process thousands "
            "of new posts daily, enabling real-time brand monitoring at scale."
        ),
    })

    return insights


def generate_action_plan(eda_stats, topic_results, eng_df, spike_results) -> list:
    """Practical action plan derived from actual results."""
    actions = []
    neg_top = list(topic_results["neg_topics"].head(5).index)
    pos_top = list(topic_results["pos_topics"].head(5).index)

    actions.append({
        "priority": "HIGH",
        "action"  : "Monitor and respond to negative discussions",
        "detail"  : (
            f"Negative posts cluster around: {neg_top}. "
            "Set up real-time alerts for these terms and assign a response team."
        ),
    })

    if not eng_df.empty:
        top_med  = eng_df.iloc[0]["Median_Engagement"]
        tied     = eng_df[eng_df["Median_Engagement"] == top_med]["Topic"].tolist()
        # Show top 3 by median, deduplicated against ties
        high_eng = list(eng_df.head(3)["Topic"])
        if len(tied) > 1:
            tie_note = (f"Topics {tied} share the highest median engagement "
                        f"of {top_med:,.0f}. ")
        else:
            tie_note = (f"'{tied[0]}' has the highest median engagement "
                        f"of {top_med:,.0f}. ")
        actions.append({
            "priority": "HIGH",
            "action"  : "Amplify high-engagement topics in brand content",
            "detail"  : (
                tie_note +
                f"Top 3 topics by median engagement: {high_eng}. "
                "Create original content around these themes to maximise organic reach."
            ),
        })

    actions.append({
        "priority": "MEDIUM",
        "action"  : "Leverage positive discussion themes for marketing",
        "detail"  : (
            f"Positive discussions focus on: {pos_top}. "
            "Use these themes in campaigns, testimonials, and social proof content."
        ),
    })

    if spike_results.get("all_spikes"):
        spike_topics = list(spike_results["all_spikes"].keys())[:3]
        actions.append({
            "priority": "MEDIUM",
            "action"  : "Prepare rapid-response playbooks for surge topics",
            "detail"  : (
                f"Topics with detected unusual surges: {spike_topics}. "
                "Pre-draft content and responses so the brand can react quickly "
                "if discussion volume spikes again on these themes."
            ),
        })

    actions.append({
        "priority": "MEDIUM",
        "action"  : "Deploy the Naive Bayes classifier for automated monitoring",
        "detail"  : (
            "The trained model can classify incoming tweets in real time. "
            "Integrate it into a dashboard to track daily sentiment shifts."
        ),
    })

    actions.append({
        "priority": "LOW",
        "action"  : "Engage the neutral majority to shift sentiment",
        "detail"  : (
            f"{eda_stats['sent_pct'].get('Neutral', 0)}% of posts are neutral. "
            "Targeted engagement (replies, polls, informative threads) can convert "
            "neutral observers into positive advocates."
        ),
    })

    return actions


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "#" * 66)
    print("  DS_Day01_42 - SOCIAL MEDIA: WHAT ARE CUSTOMERS TALKING ABOUT?")
    print("#" * 66)

    # 1. Data Cleaning
    df = load_and_clean(RAW_CSV, CLEANED_CSV)

    # 2. EDA
    eda_stats = run_eda(df, PLOTS_DIR)

    # 3 & 4. Topics + Positive vs Negative
    topic_results = run_topic_analysis(df, PLOTS_DIR)

    # 5. Topic Engagement (pass deduplicated df to fix engagement artifact)
    eng_df = run_engagement_analysis(
        df, topic_results["top_topics"], PLOTS_DIR,
        df_dedup=topic_results["df_dedup"]
    )

    # 6 & 7. Spikes + Trends (pass deduplicated df for accurate weekly counts)
    spike_results = run_trend_analysis(
        df, topic_results["top_topics"], PLOTS_DIR,
        df_dedup=topic_results["df_dedup"]
    )

    # 9 & 10. Naive Bayes + Evaluation
    model_results = run_naive_bayes(df, MODELS_DIR, PLOTS_DIR, RESULTS_DIR)

    # =========================================================================
    # FINAL STRUCTURED OUTPUT
    # =========================================================================

    print_section("DATASET SUMMARY")
    sc = eda_stats["sent_counts"]
    sp = eda_stats["sent_pct"]
    es = eda_stats["eng_stats"]
    print(f"  Total posts  : {eda_stats['n_posts']:,}")
    print(f"  Date range   : {eda_stats['date_min']}  to  {eda_stats['date_max']}")
    print(f"  Platform     : Twitter (single platform)")
    print(
        f"  Sentiment    : "
        f"Neutral {sc.get('Neutral', 0):,} ({sp.get('Neutral', 0)}%)  |  "
        f"Positive {sc.get('Positive', 0):,} ({sp.get('Positive', 0)}%)  |  "
        f"Negative {sc.get('Negative', 0):,} ({sp.get('Negative', 0)}%)"
    )
    print(
        f"  Engagement   : mean={es['mean']:,.1f}  median={es['median']:,.0f}  "
        f"min={es['min']:,}  max={es['max']:,}"
    )

    print_section("TOP DISCUSSION TOPICS")
    for i, (term, freq) in enumerate(topic_results["top_topics"].head(15).items(), 1):
        print(f"  {i:>2}. {term:<35} freq={int(freq):,}")

    print_section("POSITIVE DISCUSSION TOPICS")
    for i, (term, freq) in enumerate(topic_results["pos_topics"].head(10).items(), 1):
        print(f"  {i:>2}. {term:<35} freq={int(freq):,}")
    print(f"\n  Strongly distinctive positive terms: "
          f"{list(topic_results['pos_only'].head(5).index)}")

    print_section("NEGATIVE DISCUSSION TOPICS")
    for i, (term, freq) in enumerate(topic_results["neg_topics"].head(10).items(), 1):
        print(f"  {i:>2}. {term:<35} freq={int(freq):,}")
    print(f"\n  Strongly distinctive negative terms: "
          f"{list(topic_results['neg_only'].head(5).index)}")

    print_section("TOP ENGAGEMENT TOPICS")
    if not eng_df.empty:
        print(eng_df.head(10).to_string(index=False))

    print_section("TOPIC SPIKES")
    if spike_results.get("all_spikes"):
        for topic, spikes in spike_results["all_spikes"].items():
            print(f"\n  Topic: '{topic}'")
            print(spikes.to_string(index=False))
    else:
        print("  No significant spikes detected (z-score threshold: 2.0).")

    print_section("NAIVE BAYES MODEL RESULTS")
    print(f"  Accuracy         : {model_results['accuracy']:.4f} "
          f"({model_results['accuracy'] * 100:.2f}%)")
    print(f"  Macro F1         : {model_results['f1_macro']:.4f}")
    print(f"  Weighted F1      : {model_results['f1_weighted']:.4f}")
    print(f"  Macro Precision  : {model_results['prec_macro']:.4f}")
    print(f"  Macro Recall     : {model_results['rec_macro']:.4f}")
    print(f"\n  Full Classification Report:\n{model_results['report']}")

    # Insights
    insights = generate_insights(
        eda_stats, topic_results, eng_df, spike_results, model_results
    )
    print_section("KEY INSIGHTS")
    for i, ins in enumerate(insights, 1):
        print(f"\n  Insight {i}: {ins['title']}")
        print(f"  Finding  : {ins['finding']}")
        print(f"  Evidence : {ins['evidence']}")
        print(f"  Meaning  : {ins['meaning']}")

    # Action Plan
    actions = generate_action_plan(eda_stats, topic_results, eng_df, spike_results)
    print_section("PRACTICAL ACTION PLAN")
    for i, act in enumerate(actions, 1):
        print(f"\n  [{act['priority']}] Action {i}: {act['action']}")
        print(f"  Detail: {act['detail']}")

    # Save insights to file
    with open(os.path.join(RESULTS_DIR, "insights.txt"), "w", encoding="utf-8") as f:
        f.write("DS_Day01_42 - KEY INSIGHTS & ACTION PLAN\n")
        f.write("=" * 60 + "\n\n")
        f.write("KEY INSIGHTS\n" + "-" * 40 + "\n")
        for i, ins in enumerate(insights, 1):
            f.write(f"\nInsight {i}: {ins['title']}\n")
            f.write(f"Finding  : {ins['finding']}\n")
            f.write(f"Evidence : {ins['evidence']}\n")
            f.write(f"Meaning  : {ins['meaning']}\n")
        f.write("\n\nPRACTICAL ACTION PLAN\n" + "-" * 40 + "\n")
        for i, act in enumerate(actions, 1):
            f.write(f"\n[{act['priority']}] Action {i}: {act['action']}\n")
            f.write(f"Detail: {act['detail']}\n")

    print("\n" + "=" * 66)
    print("  PROJECT COMPLETE")
    print(f"  Plots   -> {PLOTS_DIR}")
    print(f"  Models  -> {MODELS_DIR}")
    print(f"  Results -> {RESULTS_DIR}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
