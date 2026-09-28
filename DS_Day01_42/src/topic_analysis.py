"""
topic_analysis.py
-----------------
Identifies meaningful discussion TOPICS from cleaned tweet text.
Produces MAIN VIZ 1 (Sentiment Distribution) and MAIN VIZ 2 (Top Topics).

METHODOLOGY — how topics are selected (step by step):
=======================================================

Step 1 — Deduplicate on Cleaned_Text.
  A viral tweet retweeted N times counts as ONE topic signal.
  We keep the highest-engagement representative per unique text.

Step 2 — Extract raw candidate frequencies.
  CountVectorizer is run separately for bigrams (ngram_range=(2,2))
  and unigrams (ngram_range=(1,1)) on the deduplicated texts.
  An extended stopword list removes English stopwords, Twitter noise,
  and generic filler words.

Step 3 — Bigram-preference promotion.
  For every bigram that passes the junk filter, check whether its
  component words appear as standalone unigrams in the candidate list.
  If the bigram frequency is at least MIN_BIGRAM_SUPPORT (absolute count),
  the bigram is PROMOTED: it replaces its component unigrams in the
  final ranking. The bigram's score is set to the sum of its component
  unigram frequencies (so it inherits their combined weight) if that
  sum is larger than the raw bigram count, otherwise the raw bigram
  count is used.
  This ensures "climate change" (30,141) replaces "climate" (36,816)
  and "change" (31,423) as separate entries.

Step 4 — Hashtag-form merging.
  Hashtag concatenations like "climatechange" are treated as aliases
  of their bigram form "climate change". If the bigram is already
  accepted, the hashtag unigram is suppressed.

Step 5 — Reversed-duplicate blocking.
  Before accepting a candidate, its word-set is compared to all
  already-accepted terms. A candidate whose words are a permutation
  of an accepted term (e.g. "change climate" vs "climate change")
  is rejected.

Step 6 — Canonical root-word deduplication.
  Any remaining candidate that introduces no new root word (length >= 4)
  beyond what is already covered by accepted terms is rejected.
  This prevents near-synonyms from filling the list.

Step 7 — Accept up to max_topics candidates that pass all filters.
  Topics are derived entirely from the data — nothing is hardcoded.
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
)
from data_cleaning import TWITTER_NOISE


# ── Extended stopword list ────────────────────────────────────────────────────
STOP_WORDS = set(ENGLISH_STOP_WORDS) | TWITTER_NOISE | {
    "just", "get", "got", "one", "will", "can", "us", "new",
    "make", "know", "think", "say", "said", "going", "people",
    "time", "year", "years", "day", "days", "way", "need",
    "want", "use", "used", "using", "also", "even", "still",
    "back", "come", "look", "good", "great", "really", "much",
    "many", "well", "right", "now", "today", "dont", "doesnt",
    "isnt", "arent", "wasnt", "werent", "wont", "wouldnt",
    "shouldnt", "couldnt", "didnt", "hasnt", "havent", "hadnt",
    "im", "ive", "id", "ill", "youre", "youve", "youll",
    "hes", "shes", "its", "weve", "theyre", "theyve",
    "thats", "whats", "whos", "hows", "lets",
    "lol", "omg", "oh", "ok", "yes", "no", "hi", "hey",
    "please", "thank", "thanks", "sorry", "help",
    "deal", "deals", "come", "came",
    "thing", "things", "lot", "lots", "big", "small", "old",
    "long", "high", "low", "last", "first", "next", "every",
    "never", "always", "maybe", "might", "must", "shall",
    "put", "set", "keep", "let", "run", "try", "ask",
    "show", "tell", "feel", "seem", "turn", "start", "stop",
    "true", "false", "real", "fact", "data", "report",
    "news", "media", "post", "video", "photo", "image",
    "read", "watch", "listen", "check", "see",
    "world", "country", "countries", "government", "president",
    "person", "man", "woman", "men", "women",
    "month", "week", "hour", "minute", "second",
    "number", "percent", "million", "billion", "thousand",
    # standalone words too generic to be a topic on their own
    "don", "doesn", "isn", "ve", "does", "work", "issue",
    "issues", "talk", "latest", "impact", "impacts",
    # generic verbs and common words that are not discussion themes
    # NOTE: "change" is intentionally excluded — it is a component of
    # the bigram "climate change" and must remain visible to CountVectorizer
    # so the bigram can be detected and promoted.
    "says", "believe", "believes", "believed",
    "human", "humans",
    "action", "actions",
    "fight", "fights", "fighting", "fought",
    "take", "takes", "taken", "took",
    "help", "helps", "helped",
    "live", "living",
    "save", "saves", "saving",
    "protect", "protects", "protecting",
    "support", "supports", "supporting",
    "cause", "causes", "caused",
    "affect", "affects", "affecting",
    "happen", "happens", "happening",
    "mean", "means", "meant",
    "call", "calls", "called",
    "leave", "leaves", "left",
    "face", "faces", "facing",
    "bring", "brings", "brought",
    "lead", "leads", "leading",
    "create", "creates", "created",
    "increase", "increases", "increasing",
    "reduce", "reduces", "reducing",
    "rise", "rises", "rising",
    "grow", "grows", "growing",
    "build", "builds", "building",
    "pay", "pays", "paid",
    "give", "gives", "given",
    "find", "finds", "found",
    "lose", "loses", "losing",
    "add", "adds", "added",
    "share", "shares", "shared",
    "sign", "signs", "signed",
    "join", "joins", "joined",
    "vote", "votes", "voted",
    "agree", "agrees", "agreed",
    "deny", "denies", "denied",
    "claim", "claims", "claimed",
    "argue", "argues", "argued",
    "note", "notes", "noted",
    "include", "includes", "included",
    "continue", "continues", "continued",
    "remain", "remains", "remained",
    "become", "becomes", "became",
    "allow", "allows", "allowed",
    "require", "requires", "required",
    "consider", "considers", "considered",
    "expect", "expects", "expected",
    "plan", "plans", "planned",
    "fail", "fails", "failed",
    "pass", "passes", "passed",
    "begin", "begins", "began",
}

# Minimum number of unique posts a bigram must appear in to be promoted
# over its component unigrams. Set conservatively so rare bigrams don't
# displace common unigrams.
MIN_BIGRAM_SUPPORT = 150

# ── Junk-phrase patterns ──────────────────────────────────────────────────────
_JUNK_PATTERNS = [
    re.compile(r"^\d+\s+\w+$"),       # "19 months"
    re.compile(r"^\w+\s+\d+$"),       # "months 19"
    re.compile(r"^\d{4}\s+\d{4}$"),   # "2014 2015"
    re.compile(r"^\d{4}\s+\w+$"),     # "2018 pace"
    re.compile(r"^\w+\s+\d{4}$"),     # "history 2014"
    re.compile(r"^\d+$"),             # pure numbers
    re.compile(r"^[a-z]\s+[a-z]$"),   # single-letter pairs "e u"
    re.compile(r"^\w+\s+isn$"),       # "change isn"
    re.compile(r"^\w+\s+doesn$"),     # trailing fragment
    re.compile(r"^don\s+\w+$"),       # "don believe"
    re.compile(r"^\w+\s+chan$"),      # "climate chan" (truncated)
]


def _is_junk(term: str) -> bool:
    for pat in _JUNK_PATTERNS:
        if pat.match(term.strip()):
            return True
    return False


def _word_set(term: str) -> frozenset:
    """Frozenset of all words — used to detect reversed duplicates."""
    return frozenset(term.split())


def _root_words(term: str) -> set:
    """Words of length >= 4 — used for canonical deduplication."""
    return {w for w in term.split() if len(w) >= 4}


def _get_raw_freq(texts: pd.Series, ngram_range=(1, 2),
                  max_features=20000, min_df=3) -> pd.Series:
    """Count-based frequency. Returns all candidates ranked by frequency."""
    vec = CountVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
    )
    X      = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).flatten()
    terms  = vec.get_feature_names_out()
    return pd.Series(counts, index=terms).sort_values(ascending=False)


def _get_top_tfidf(texts: pd.Series, n: int = 10,
                   ngram_range=(1, 2)) -> pd.Series:
    """TF-IDF mean score per term. Used for sentiment distinctiveness."""
    vec = TfidfVectorizer(
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        max_features=10000,
        min_df=3,
        sublinear_tf=True,
    )
    X           = vec.fit_transform(texts)
    mean_scores = np.asarray(X.mean(axis=0)).flatten()
    terms       = vec.get_feature_names_out()
    return pd.Series(mean_scores, index=terms).sort_values(ascending=False).head(n)


def _build_topic_list(raw_bigrams: pd.Series,
                      raw_unigrams: pd.Series,
                      max_topics: int = 15) -> pd.Series:
    """
    Builds the final topic list using bigram-preference promotion.

    Algorithm
    ---------
    1. For each bigram with freq >= MIN_BIGRAM_SUPPORT:
         - Check if both component words appear as unigrams.
         - If yes, mark those unigrams as "absorbed" and assign the
           bigram a promoted score = max(bigram_freq, sum of component
           unigram freqs). This makes the bigram rank at least as high
           as its components would have individually.
    2. Build a merged candidate list:
         - All bigrams (junk-filtered, with promoted scores where applicable)
         - All unigrams that were NOT absorbed by a bigram
         - Sort by score descending
    3. Iterate through merged candidates:
         - Skip junk
         - Skip reversed duplicates of already-accepted terms
         - Skip hashtag-form aliases of already-accepted bigrams
           (e.g. "climatechange" if "climate change" is accepted)
         - Skip if no new root word is introduced
         - Accept otherwise
    4. Return the first max_topics accepted terms.
    """
    absorbed_unigrams = set()   # unigrams subsumed by a promoted bigram
    promoted          = {}      # bigram -> promoted score

    # Process bigrams in frequency order so the highest-frequency bigram
    # wins the right to absorb each unigram word.
    # A unigram word can only be absorbed ONCE — by the first (highest-freq)
    # bigram that claims it. This prevents "climate" from being absorbed by
    # every bigram that contains it (global climate, trump climate, etc.).
    for bigram, bfreq in raw_bigrams.sort_values(ascending=False).items():
        if _is_junk(bigram):
            continue
        if bfreq < MIN_BIGRAM_SUPPORT:
            continue
        words = bigram.split()
        if len(words) != 2:
            continue
        w1, w2 = words
        # Both component words must exist as unigrams AND not yet absorbed
        w1_free = w1 in raw_unigrams.index and w1 not in absorbed_unigrams
        w2_free = w2 in raw_unigrams.index and w2 not in absorbed_unigrams
        if w1_free and w2_free:
            # Promoted score = raw bigram frequency only.
            # We do NOT sum component unigrams because that would inflate
            # scores for bigrams that share a high-frequency word like
            # "climate" across many different bigrams.
            promoted[bigram] = int(bfreq)
            absorbed_unigrams.add(w1)
            absorbed_unigrams.add(w2)
        else:
            # One or both component words already absorbed — keep raw freq
            # but do NOT absorb anything further
            promoted[bigram] = int(bfreq)

    # Unigrams not absorbed by any bigram
    surviving_unigrams = raw_unigrams[
        ~raw_unigrams.index.isin(absorbed_unigrams)
    ]

    # Merge: promoted bigrams + surviving unigrams, sort by score
    merged = pd.concat([
        pd.Series(promoted),
        surviving_unigrams,
    ]).sort_values(ascending=False)

    # Deduplicate index (bigram wins if same key somehow appears twice)
    merged = merged[~merged.index.duplicated(keep="first")]

    # ── Selection pass ────────────────────────────────────────────────────────
    accepted       = {}
    used_roots     = set()
    used_wordsets  = []   # list of frozensets for reversed-duplicate check
    accepted_bigrams_words = set()  # all individual words in accepted bigrams

    for term, score in merged.items():
        if _is_junk(term):
            continue

        words = term.split()

        # Block reversed duplicates: "change climate" if "climate change" accepted
        ws = _word_set(term)
        if ws in used_wordsets:
            continue

        # Block hashtag aliases: "climatechange" if "climate change" accepted
        # A unigram is a hashtag alias if its characters equal a bigram with
        # spaces removed that is already accepted.
        if len(words) == 1:
            no_space_forms = {
                "".join(b.split()) for b in accepted if len(b.split()) == 2
            }
            if term in no_space_forms:
                continue
            # Also block if the unigram is a substring concat of accepted bigram words
            # e.g. "globalwarming" when "global warming" is accepted
            if term in accepted_bigrams_words:
                continue

        roots = _root_words(term)
        if not roots:
            continue

        # Require at least one new root word
        if not (roots - used_roots):
            continue

        accepted[term] = int(score)
        used_roots    |= roots
        used_wordsets.append(ws)
        if len(words) == 2:
            accepted_bigrams_words.update(words)

        if len(accepted) >= max_topics:
            break

    return pd.Series(accepted)


def run_topic_analysis(df: pd.DataFrame, plots_dir: str) -> dict:
    """
    Extracts meaningful discussion topics and compares Positive vs Negative.
    Returns a dict with topic Series for downstream engagement and trend analysis.
    """
    print("\n" + "=" * 60)
    print("STEP 3 & 4 - DISCUSSION TOPICS & POSITIVE vs NEGATIVE")
    print("=" * 60)

    # ── Step 1: Deduplicate on Cleaned_Text ───────────────────────────────────
    df_dedup = (
        df.sort_values("Engagement", ascending=False)
          .drop_duplicates(subset="Cleaned_Text", keep="first")
          .reset_index(drop=True)
    )
    print(f"\nUnique tweet texts (after dedup): {len(df_dedup):,}  "
          f"(removed {len(df) - len(df_dedup):,} duplicate texts)")

    texts_dedup = df_dedup["Cleaned_Text"].fillna("")

    # ── Step 2: Raw frequencies ───────────────────────────────────────────────
    raw_bigrams  = _get_raw_freq(texts_dedup, ngram_range=(2, 2))
    raw_unigrams = _get_raw_freq(texts_dedup, ngram_range=(1, 1))

    # ── Steps 3-6: Build final topic list with bigram-preference ─────────────
    top_topics = _build_topic_list(raw_bigrams, raw_unigrams, max_topics=15)

    print("\nTop 15 Discussion Topics (bigram-preferred, deduplicated):")
    for i, (term, score) in enumerate(top_topics.items(), 1):
        n_words = len(term.split())
        kind    = "bigram" if n_words == 2 else "unigram"
        print(f"  {i:>2}. {term:<30}  score={int(score):>7,}  [{kind}]")

    # ── Per-sentiment topics ──────────────────────────────────────────────────
    pos_dedup = df_dedup.loc[df_dedup["Sentiment_Label"] == "Positive",
                             "Cleaned_Text"].fillna("")
    neg_dedup = df_dedup.loc[df_dedup["Sentiment_Label"] == "Negative",
                             "Cleaned_Text"].fillna("")

    pos_bi  = _get_raw_freq(pos_dedup, ngram_range=(2, 2))
    pos_uni = _get_raw_freq(pos_dedup, ngram_range=(1, 1))
    neg_bi  = _get_raw_freq(neg_dedup, ngram_range=(2, 2))
    neg_uni = _get_raw_freq(neg_dedup, ngram_range=(1, 1))

    pos_topics = _build_topic_list(pos_bi, pos_uni, max_topics=12)
    neg_topics = _build_topic_list(neg_bi, neg_uni, max_topics=12)

    print("\nTop 12 POSITIVE Discussion Topics:")
    for i, (term, score) in enumerate(pos_topics.items(), 1):
        print(f"  {i:>2}. {term:<30}  score={int(score):>7,}")

    print("\nTop 12 NEGATIVE Discussion Topics:")
    for i, (term, score) in enumerate(neg_topics.items(), 1):
        print(f"  {i:>2}. {term:<30}  score={int(score):>7,}")

    # TF-IDF distinctiveness
    pos_tfidf = _get_top_tfidf(pos_dedup, n=10)
    neg_tfidf = _get_top_tfidf(neg_dedup, n=10)
    pos_only  = pos_tfidf[~pos_tfidf.index.isin(neg_tfidf.index)]
    neg_only  = neg_tfidf[~neg_tfidf.index.isin(pos_tfidf.index)]

    print("\nDistinctive TF-IDF Terms:")
    print("  Strongly Positive:", list(pos_only.head(5).index))
    print("  Strongly Negative:", list(neg_only.head(5).index))

    # ------------------------------------------------------------------
    # MAIN VISUALIZATION 1: Sentiment Distribution
    # ------------------------------------------------------------------
    sent_counts = df["Sentiment_Label"].value_counts()
    colors      = {"Positive": "#2ecc71", "Neutral": "#3498db", "Negative": "#e74c3c"}
    bar_colors  = [colors.get(l, "#95a5a6") for l in sent_counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(sent_counts.index, sent_counts.values,
                  color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, sent_counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 500,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("MAIN VIZ 1 - Sentiment Distribution of Twitter Posts",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Number of Posts", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, sent_counts.max() * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "viz1_sentiment_distribution.png"), dpi=150)
    plt.close()
    print("\n[Saved] viz1_sentiment_distribution.png")

    # ------------------------------------------------------------------
    # MAIN VISUALIZATION 2: Top Discussion Topics
    # ------------------------------------------------------------------
    plot_topics = top_topics.sort_values()   # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(11, 7))
    bar_colors_t = ["#2980b9" if len(t.split()) == 2 else "#7fb3d3"
                    for t in plot_topics.index]
    bars = ax.barh(plot_topics.index, plot_topics.values,
                   color=bar_colors_t, edgecolor="white")
    for bar, val in zip(bars, plot_topics.values):
        ax.text(bar.get_width() + plot_topics.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", fontsize=9)

    # Legend to explain colour coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2980b9", label="Bigram (phrase)"),
        Patch(facecolor="#7fb3d3", label="Unigram (single word)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_title("MAIN VIZ 2 - Top Discussion Terms and Phrases on Twitter\n"
                 "(meaningful bigrams preferred over isolated words; "
                 "scores = unique post count per deduplicated corpus)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Frequency (unique posts, viral retweets deduplicated)", fontsize=11)
    ax.set_ylabel("Topic", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "viz2_top_discussion_topics.png"), dpi=150)
    plt.close()
    print("[Saved] viz2_top_discussion_topics.png")

    # ------------------------------------------------------------------
    # Diagnostic: Positive vs Negative shared topic comparison
    # ------------------------------------------------------------------
    common = set(pos_topics.head(10).index) & set(neg_topics.head(10).index)
    if len(common) >= 3:
        common_terms = sorted(common)
        pos_vals = [int(pos_topics.get(t, 0)) for t in common_terms]
        neg_vals = [int(neg_topics.get(t, 0)) for t in common_terms]
        x = range(len(common_terms))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - 0.2 for i in x], pos_vals, width=0.4,
               label="Positive", color="#2ecc71", edgecolor="white")
        ax.bar([i + 0.2 for i in x], neg_vals, width=0.4,
               label="Negative", color="#e74c3c", edgecolor="white")
        ax.set_xticks(list(x))
        ax.set_xticklabels(common_terms, rotation=30, ha="right", fontsize=9)
        ax.set_title("Diagnostic - Shared Topics: Positive vs Negative",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Topic Score (unique posts)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir,
                    "diagnostic_pos_neg_topic_comparison.png"), dpi=120)
        plt.close()
        print("[Saved] diagnostic_pos_neg_topic_comparison.png")

    return {
        "top_topics": top_topics,
        "pos_topics": pos_topics,
        "neg_topics": neg_topics,
        "pos_tfidf" : pos_tfidf,
        "neg_tfidf" : neg_tfidf,
        "pos_only"  : pos_only,
        "neg_only"  : neg_only,
        "df_dedup"  : df_dedup,
    }
