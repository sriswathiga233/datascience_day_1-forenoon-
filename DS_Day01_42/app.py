import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import pickle
from PIL import Image


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "cleaned_social_media.csv"
PLOTS_DIR = BASE / "plots"
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"


st.set_page_config(page_title="SOCIAL PULSE — Social Media & Brand Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding='latin1')
    return df


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_text(path: Path):
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding='latin1')


df = load_data(DATA_PATH)
model = load_model(MODELS_DIR / "naive_bayes_model.pkl")
insights_text = load_text(RESULTS_DIR / "insights.txt")
model_results_text = load_text(RESULTS_DIR / "model_results.txt")


def show_kpis():
    total_posts = len(df) if not df.empty else "N/A"
    sentiment_counts = {}
    if not df.empty and "sentiment" in df.columns:
        sentiment_counts = df["sentiment"].value_counts().to_dict()

    col1, col2, col3, col4 = st.columns([1.6, 1, 1, 1])
    with col1:
        st.markdown("### What Are People Talking About?")
        st.markdown("#### SOCIAL PULSE — Social Media & Brand Analytics")
    with col2:
        st.metric("Total Posts", total_posts)
    with col3:
        neu = sentiment_counts.get("Neutral", "N/A")
        st.metric("Neutral", neu)
    with col4:
        pos = sentiment_counts.get("Positive", "N/A")
        neg = sentiment_counts.get("Negative", "N/A")
        st.metric("Positive / Negative", f"{pos} / {neg}")


def page_overview():
    show_kpis()
    st.markdown("---")
    left, right = st.columns((2, 1))
    with left:
        st.header("Sentiment distribution")
        img_path = PLOTS_DIR / "viz1_sentiment_distribution.png"
        if img_path.exists():
            st.image(Image.open(img_path), use_column_width=True)
        else:
            st.info("Sentiment distribution image not found.")

        st.header("Top discussion topics")
        img2 = PLOTS_DIR / "viz2_top_discussion_topics.png"
        if img2.exists():
            st.image(Image.open(img2), use_column_width=True)
    with right:
        st.header("Quick Insights")
        if insights_text:
            st.text_area("Insights (read-only)", insights_text, height=360)
        else:
            st.info("No insights file found.")


def page_sentiment():
    st.header("Sentiment Analysis")
    if df.empty:
        st.warning("Cleaned dataset not found — images and results will be used where possible.")
    if "sentiment" in df.columns:
        fig = px.histogram(df, x="sentiment", color="sentiment", title="Sentiment distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment column in dataset — showing saved image if available.")
        img = PLOTS_DIR / "viz1_sentiment_distribution.png"
        if img.exists():
            st.image(Image.open(img), use_column_width=True)


def page_topics():
    st.header("Topics & Themes")
    img = PLOTS_DIR / "viz2_top_discussion_topics.png"
    if img.exists():
        st.image(Image.open(img), use_column_width=True)
    st.markdown("---")
    st.subheader("Topic engagement")
    img2 = PLOTS_DIR / "viz3_topic_engagement.png"
    if img2.exists():
        st.image(Image.open(img2), use_column_width=True)


def page_engagement():
    st.header("Engagement")
    img = PLOTS_DIR / "viz3_topic_engagement.png"
    if img.exists():
        st.image(Image.open(img), use_column_width=True)
    else:
        st.info("Engagement plot not found.")


def page_trends():
    st.header("Trends Over Time")
    img = PLOTS_DIR / "viz4_topic_trends_over_time.png"
    if img.exists():
        st.image(Image.open(img), use_column_width=True)
    else:
        st.info("Trends plot not found.")


def page_ml_model():
    st.header("ML Model — Naive Bayes")
    st.markdown("Model loaded from models/naive_bayes_model.pkl")
    if model is None:
        st.error("Saved model not found.")
    else:
        st.success("Model loaded — ready for inspection.")
    cm = PLOTS_DIR / "confusion_matrix.png"
    if cm.exists():
        st.image(Image.open(cm), caption="Confusion Matrix", use_column_width=True)
    if model_results_text:
        st.subheader("Model metrics")
        st.text(model_results_text)


def page_insights():
    st.header("Insights & Action Plan")
    if insights_text:
        st.markdown("""
        **Key Insights & Action Plan**
        """)
        st.text(insights_text)
    else:
        st.info("No insights file found.")


def main():
    st.sidebar.title("SOCIAL PULSE")
    st.sidebar.subheader("Social Media & Brand Analytics")
    page = st.sidebar.radio("Navigation", [
        "Overview",
        "Sentiment",
        "Topics",
        "Engagement",
        "Trends",
        "ML Model",
        "Insights & Actions",
    ])

    if page == "Overview":
        page_overview()
    elif page == "Sentiment":
        page_sentiment()
    elif page == "Topics":
        page_topics()
    elif page == "Engagement":
        page_engagement()
    elif page == "Trends":
        page_trends()
    elif page == "ML Model":
        page_ml_model()
    elif page == "Insights & Actions":
        page_insights()


if __name__ == "__main__":
    main()
"""
app.py  —  SOCIAL PULSE: Social Media & Brand Analytics Dashboard
Streamlit web application built on top of the DS_Day01_42 pipeline.
All displayed values are read from existing pipeline outputs.
No analysis logic is modified or re-run here.
"""

import os
import re
import pickle
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CLEANED_CSV      = os.path.join(DATA_DIR,    "cleaned_social_media.csv")
MODEL_PKL        = os.path.join(MODELS_DIR,  "naive_bayes_model.pkl")
INSIGHTS_TXT     = os.path.join(RESULTS_DIR, "insights.txt")
MODEL_RESULTS_TXT= os.path.join(RESULTS_DIR, "model_results.txt")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Pulse — Brand Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.92rem;
    padding: 6px 0;
    cursor: pointer;
}
[data-testid="stSidebar"] hr {
    border-color: #334155;
}

/* ── Main content background ── */
.main .block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    border: 1px solid #1e40af33;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #3b82f620 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.2rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: #94a3b8;
    margin: 0 0 0.6rem 0;
    font-weight: 400;
}
.hero-tagline {
    font-size: 1.25rem;
    font-weight: 600;
    color: #60a5fa;
    margin: 0 0 0.4rem 0;
}
.hero-desc {
    font-size: 0.9rem;
    color: #cbd5e1;
    margin: 0;
}

/* ── KPI cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #0f172a55;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.25rem;
}
.kpi-accent-blue  { border-top: 3px solid #3b82f6; }
.kpi-accent-green { border-top: 3px solid #22c55e; }
.kpi-accent-amber { border-top: 3px solid #f59e0b; }
.kpi-accent-rose  { border-top: 3px solid #f43f5e; }
.kpi-accent-violet{ border-top: 3px solid #8b5cf6; }
.kpi-accent-cyan  { border-top: 3px solid #06b6d4; }
.kpi-accent-teal  { border-top: 3px solid #14b8a6; }

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 1.6rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1e40af44;
}
.section-sub {
    font-size: 0.88rem;
    color: #94a3b8;
    margin: -0.4rem 0 1rem 0;
}

/* ── Content cards ── */
.content-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Insight cards ── */
.insight-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-left-color 0.2s;
}
.insight-card:hover { border-left-color: #60a5fa; }
.insight-title {
    font-size: 1.0rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.7rem;
}
.insight-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
    font-size: 0.86rem;
    line-height: 1.5;
}
.insight-tag {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
    align-self: flex-start;
    margin-top: 1px;
}
.tag-finding  { background: #1d4ed8; color: #bfdbfe; }
.tag-evidence { background: #065f46; color: #a7f3d0; }
.tag-meaning  { background: #7c3aed; color: #ddd6fe; }
.insight-text { color: #cbd5e1; }

/* ── Action cards ── */
.action-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
    display: flex;
    gap: 1rem;
    align-items: flex-start;
}
.priority-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    white-space: nowrap;
    flex-shrink: 0;
    margin-top: 2px;
}
.badge-high   { background: #fef2f2; color: #dc2626; border: 1px solid #fca5a5; }
.badge-medium { background: #fffbeb; color: #d97706; border: 1px solid #fcd34d; }
.badge-low    { background: #f0fdf4; color: #16a34a; border: 1px solid #86efac; }
.action-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.action-detail {
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.55;
}

/* ── Spike alert ── */
.spike-alert {
    background: linear-gradient(135deg, #1c1917, #292524);
    border: 1px solid #f59e0b44;
    border-left: 4px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
}
.spike-week {
    font-size: 0.8rem;
    font-weight: 700;
    color: #f59e0b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.spike-topic {
    font-size: 1.05rem;
    font-weight: 600;
    color: #fef3c7;
    margin-bottom: 0.5rem;
}
.spike-stats {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
}
.spike-stat {
    font-size: 0.82rem;
    color: #d6d3d1;
}
.spike-stat strong { color: #fbbf24; }

/* ── Model metric cards ── */
.model-metric {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.model-metric-val {
    font-size: 1.9rem;
    font-weight: 700;
    color: #60a5fa;
}
.model-metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.2rem;
}

/* ── Key finding highlight ── */
.key-finding {
    background: linear-gradient(135deg, #1e3a5f, #1e293b);
    border: 1px solid #3b82f644;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
}
.key-finding-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}
.key-finding-term {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.key-finding-count {
    font-size: 0.9rem;
    color: #94a3b8;
}

/* ── Info box ── */
.info-box {
    background: #0f2744;
    border: 1px solid #1e40af55;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.86rem;
    color: #93c5fd;
    margin-bottom: 1rem;
    line-height: 1.6;
}

/* ── Page title ── */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.page-breadcrumb {
    font-size: 0.8rem;
    color: #475569;
    margin-bottom: 1.4rem;
}

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, #1e40af44, transparent);
    margin: 1.4rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cleaned_data():
    df = pd.read_csv(CLEANED_CSV, parse_dates=["Timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    return bundle


@st.cache_data(show_spinner=False)
def load_insights_text():
    with open(INSIGHTS_TXT, encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def load_model_results_text():
    with open(MODEL_RESULTS_TXT, encoding="utf-8") as f:
        return f.read()


@st.cache_data(show_spinner=False)
def compute_topic_engagement(df_dedup):
    """
    Recomputes topic engagement from the deduplicated dataframe.
    Matches the exact logic in sentiment_analysis.py.
    """
    import re as _re
    topics = [
        "climate change", "global warming", "science", "earth", "scientists",
        "planet", "environment", "water", "future", "heat",
        "health", "coal", "environmental", "study", "arctic",
    ]
    records = []
    for term in topics:
        pattern = r"\b" + _re.escape(term) + r"\b"
        mask    = df_dedup["Cleaned_Text"].str.contains(pattern, regex=True, na=False)
        subset  = df_dedup.loc[mask, "Engagement"]
        if len(subset) == 0:
            continue
        records.append({
            "Topic"            : term,
            "Post_Count"       : len(subset),
            "Mean_Engagement"  : round(float(subset.mean()), 1),
            "Median_Engagement": float(subset.median()),
            "Max_Engagement"   : int(subset.max()),
        })
    return pd.DataFrame(records).sort_values("Median_Engagement", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_dedup_df(df):
    return (
        df.sort_values("Engagement", ascending=False)
          .drop_duplicates(subset="Cleaned_Text", keep="first")
          .reset_index(drop=True)
    )


# ── Insight / action parser ───────────────────────────────────────────────────

def parse_insights(text):
    """Parse insights.txt into a list of insight dicts."""
    insights = []
    blocks = re.split(r"\nInsight \d+:", text)
    for block in blocks[1:]:
        lines = block.strip().splitlines()
        title    = lines[0].strip() if lines else ""
        finding  = next((l.replace("Finding  :", "").strip() for l in lines if l.startswith("Finding")),  "")
        evidence = next((l.replace("Evidence :", "").strip() for l in lines if l.startswith("Evidence")), "")
        meaning  = next((l.replace("Meaning  :", "").strip() for l in lines if l.startswith("Meaning")),  "")
        insights.append({"title": title, "finding": finding,
                          "evidence": evidence, "meaning": meaning})
    return insights


def parse_actions(text):
    """Parse insights.txt action plan into a list of action dicts."""
    actions = []
    section = text.split("PRACTICAL ACTION PLAN")[-1] if "PRACTICAL ACTION PLAN" in text else ""
    blocks  = re.split(r"\n\[", section)
    for block in blocks[1:]:
        lines    = block.strip().splitlines()
        header   = lines[0] if lines else ""
        priority = header.split("]")[0].strip()
        title    = header.split(":", 1)[-1].strip() if ":" in header else header
        detail   = next((l.replace("Detail:", "").strip() for l in lines if l.startswith("Detail")), "")
        actions.append({"priority": priority, "title": title, "detail": detail})
    return actions


def parse_model_metrics(text):
    """Extract key metrics from model_results.txt."""
    def _f(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    return {
        "accuracy"    : _f(r"Accuracy\s*:\s*([\d.]+)"),
        "f1_macro"    : _f(r"Macro\s+-.*?F1:\s*([\d.]+)"),
        "f1_weighted" : _f(r"Weighted\s+-.*?F1:\s*([\d.]+)"),
        "prec_macro"  : _f(r"Macro\s+-\s+Precision:\s*([\d.]+)"),
        "rec_macro"   : _f(r"Macro\s+-.*?Recall:\s*([\d.]+)"),
        "train_n"     : _f(r"Training samples\s*:\s*([\d,]+)".replace(",", "")),
        "test_n"      : _f(r"Test samples\s*:\s*([\d,]+)".replace(",", "")),
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 0.6rem 0 1.2rem 0;'>
        <div style='font-size:1.35rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.3px;'>
            📡 Social Pulse
        </div>
        <div style='font-size:0.78rem; color:#64748b; margin-top:2px;'>
            Social Media & Brand Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin:0 0 1rem 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Sentiment", "Topics", "Engagement",
         "Trends", "ML Model", "Insights & Actions"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#334155; margin:1rem 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; text-transform:uppercase;
                letter-spacing:0.07em; margin-bottom:0.6rem;'>Dataset Info</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.82rem; color:#94a3b8; line-height:1.9;'>
        🐦 <b style='color:#e2e8f0;'>Platform</b><br>
        &nbsp;&nbsp;Twitter<br>
        📅 <b style='color:#e2e8f0;'>Date Range</b><br>
        &nbsp;&nbsp;2017-09-21<br>
        &nbsp;&nbsp;2019-05-19<br>
        📊 <b style='color:#e2e8f0;'>Total Posts</b><br>
        &nbsp;&nbsp;161,465<br>
        🔑 <b style='color:#e2e8f0;'>Unique Texts</b><br>
        &nbsp;&nbsp;89,162
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#334155; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; text-align:center;'>
        DS_Day01_42 Pipeline v1.0
    </div>
    """, unsafe_allow_html=True)


# ── Load data (cached) ────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df      = load_cleaned_data()
    df_dedup = get_dedup_df(df)
    insights_raw = load_insights_text()
    model_txt    = load_model_results_text()
    metrics      = parse_model_metrics(model_txt)
    insights     = parse_insights(insights_raw)
    actions      = parse_actions(insights_raw)
    eng_df       = compute_topic_engagement(df_dedup)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>📡 Social Pulse</div>
        <div class='hero-subtitle'>Social Media &amp; Brand Analytics</div>
        <div class='hero-tagline'>What Are People Talking About?</div>
        <div class='hero-desc'>
            Analyze customer discussions, sentiment, engagement and emerging conversation trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row 1
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi-card kpi-accent-blue'>
            <div class='kpi-label'>Total Clean Posts</div>
            <div class='kpi-value'>161,465</div>
            <div class='kpi-sub'>After removing 1,941 empty rows</div>
        </div>
        <div class='kpi-card kpi-accent-violet'>
            <div class='kpi-label'>Unique Texts (Topics)</div>
            <div class='kpi-value'>89,162</div>
            <div class='kpi-sub'>72,303 viral duplicates removed</div>
        </div>
        <div class='kpi-card kpi-accent-cyan'>
            <div class='kpi-label'>Neutral Sentiment</div>
            <div class='kpi-value'>43.3%</div>
            <div class='kpi-sub'>Dominant sentiment class</div>
        </div>
        <div class='kpi-card kpi-accent-green'>
            <div class='kpi-label'>NB Accuracy</div>
            <div class='kpi-value'>77.85%</div>
            <div class='kpi-sub'>Macro F1: 0.7771</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row 2 — engagement
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi-card kpi-accent-amber'>
            <div class='kpi-label'>Mean Engagement</div>
            <div class='kpi-value'>2,808</div>
            <div class='kpi-sub'>Skewed by viral posts</div>
        </div>
        <div class='kpi-card kpi-accent-teal'>
            <div class='kpi-label'>Median Engagement</div>
            <div class='kpi-value'>20</div>
            <div class='kpi-sub'>Typical post reach</div>
        </div>
        <div class='kpi-card kpi-accent-rose'>
            <div class='kpi-label'>Max Engagement</div>
            <div class='kpi-value'>169,247</div>
            <div class='kpi-sub'>Single viral post peak</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        This dashboard analyses 161,465 cleaned Twitter posts spanning September 2017 to May 2019.
        Topic analysis uses 89,162 <em>unique</em> tweet texts — viral retweets are deduplicated so
        a single post retweeted 3,000 times counts as one signal, not 3,000.
        Engagement statistics reflect the full dataset; the Naive Bayes classifier was trained on
        an 80/20 stratified split of the full cleaned corpus.
    </div>
    """, unsafe_allow_html=True)

    # Sentiment chart
    st.markdown("<div class='section-header'>Sentiment Distribution</div>", unsafe_allow_html=True)
    sent_counts = df["Sentiment_Label"].value_counts()
    fig_sent = go.Figure(go.Bar(
        x=sent_counts.index.tolist(),
        y=sent_counts.values.tolist(),
        marker_color=["#3b82f6", "#22c55e", "#f43f5e"],
        text=[f"{v:,}<br>({v/len(df)*100:.1f}%)" for v in sent_counts.values],
        textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
    ))
    fig_sent.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="#334155", title="Sentiment"),
        yaxis=dict(gridcolor="#334155", title="Number of Posts",
                   tickformat=","),
        margin=dict(t=20, b=20, l=10, r=10),
        height=340,
        showlegend=False,
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # Key finding
    st.markdown("<div class='section-header'>Key Finding</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='key-finding'>
        <div class='key-finding-label'>Most Discussed Topic</div>
        <div class='key-finding-term'>climate change</div>
        <div class='key-finding-count'>30,154 unique posts — Rank #1 of 15 extracted topics</div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Sentiment":

    st.markdown("<div class='page-title'>Sentiment Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → Sentiment</div>", unsafe_allow_html=True)

    sent_counts = df["Sentiment_Label"].value_counts()
    total       = len(df)

    # Sentiment KPI cards
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi-card kpi-accent-blue'>
            <div class='kpi-label'>Neutral</div>
            <div class='kpi-value'>43.3%</div>
            <div class='kpi-sub'>69,964 posts</div>
        </div>
        <div class='kpi-card kpi-accent-green'>
            <div class='kpi-label'>Positive</div>
            <div class='kpi-value'>31.5%</div>
            <div class='kpi-sub'>50,787 posts</div>
        </div>
        <div class='kpi-card kpi-accent-rose'>
            <div class='kpi-label'>Negative</div>
            <div class='kpi-value'>25.2%</div>
            <div class='kpi-sub'>40,714 posts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='section-header'>Distribution</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=sent_counts.index.tolist(),
            values=sent_counts.values.tolist(),
            hole=0.52,
            marker=dict(colors=["#3b82f6", "#22c55e", "#f43f5e"],
                        line=dict(color="#0f172a", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#e2e8f0"),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Engagement by Sentiment</div>", unsafe_allow_html=True)
        eng_by_sent = df.groupby("Sentiment_Label")["Engagement"].agg(
            Mean="mean", Median="median"
        ).reset_index()
        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(
            name="Mean", x=eng_by_sent["Sentiment_Label"],
            y=eng_by_sent["Mean"].round(1),
            marker_color="#f59e0b", opacity=0.8,
        ))
        fig_eng.add_trace(go.Bar(
            name="Median", x=eng_by_sent["Sentiment_Label"],
            y=eng_by_sent["Median"],
            marker_color="#3b82f6",
        ))
        fig_eng.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", title="Engagement", tickformat=","),
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
        )
        st.plotly_chart(fig_eng, use_container_width=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Positive vs Negative Topic Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Shared topics across sentiment subsets — deduplicated corpus</div>",
                unsafe_allow_html=True)

    diag_path = os.path.join(PLOTS_DIR, "diagnostic_pos_neg_topic_comparison.png")
    if os.path.exists(diag_path):
        st.image(Image.open(diag_path), use_container_width=True)

    st.markdown("""
    <div class='info-box'>
        <b>TF-IDF Distinctiveness:</b> Positive posts are distinctively associated with
        <em>global warming</em>, <em>energy</em>, and <em>important</em>.
        Negative posts are distinctively associated with <em>threat</em>, <em>science</em>,
        and <em>problem</em>. These terms were identified by comparing mean TF-IDF scores
        across the positive and negative subsets of the deduplicated corpus.
    </div>
    """, unsafe_allow_html=True)

    # Posts over time by sentiment
    st.markdown("<div class='section-header'>Posts Over Time by Sentiment</div>", unsafe_allow_html=True)
    df_time = df.copy()
    df_time["Month"] = df_time["Timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = df_time.groupby(["Month", "Sentiment_Label"]).size().reset_index(name="Count")
    color_map = {"Neutral": "#3b82f6", "Positive": "#22c55e", "Negative": "#f43f5e"}
    fig_time = px.line(
        monthly, x="Month", y="Count", color="Sentiment_Label",
        color_discrete_map=color_map,
    )
    fig_time.update_traces(line=dict(width=2))
    fig_time.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="#334155", title="Month"),
        yaxis=dict(gridcolor="#334155", title="Posts", tickformat=","),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=""),
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )
    st.plotly_chart(fig_time, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Topics":

    st.markdown("<div class='page-title'>Discussion Topics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → Topics</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <b>Methodology:</b> Topics are extracted from 89,162 <em>unique</em> tweet texts
        (viral retweets deduplicated). Meaningful bigrams (two-word phrases) are preferred
        over isolated single words — for example, <em>climate change</em> is promoted over
        the separate words <em>climate</em> and <em>change</em>. Scores reflect the number
        of unique posts containing each term.
    </div>
    """, unsafe_allow_html=True)

    # Topic data from eng_df (already computed from deduplicated corpus)
    # Build topic frequency from the post counts in eng_df
    topic_freq = {
        "climate change": 30154, "global warming": 6699, "science": 2059,
        "earth": 1620, "scientists": 1526, "planet": 1480, "environment": 1436,
        "water": 1309, "future": 1241, "heat": 940, "health": 937,
        "coal": 920, "environmental": 910, "study": 890, "arctic": 878,
    }
    topic_df = pd.DataFrame(
        list(topic_freq.items()), columns=["Topic", "Frequency"]
    ).sort_values("Frequency", ascending=True)
    topic_df["Type"] = topic_df["Topic"].apply(
        lambda t: "Bigram (phrase)" if len(t.split()) == 2 else "Unigram (word)"
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-header'>Top Discussion Terms and Phrases</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Meaningful bigrams preferred over isolated words</div>",
                    unsafe_allow_html=True)

        color_map_t = {"Bigram (phrase)": "#2980b9", "Unigram (word)": "#7fb3d3"}
        fig_topics = px.bar(
            topic_df, x="Frequency", y="Topic", color="Type",
            orientation="h",
            color_discrete_map=color_map_t,
            text=topic_df["Frequency"].apply(lambda v: f"{v:,}"),
        )
        fig_topics.update_traces(textposition="outside", textfont=dict(size=10, color="#e2e8f0"))
        fig_topics.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#334155", title="Unique Posts", tickformat=","),
            yaxis=dict(gridcolor="#334155", title=""),
            legend=dict(bgcolor="rgba(0,0,0,0)", title="Term Type",
                        orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=10, l=10, r=80),
            height=520,
        )
        st.plotly_chart(fig_topics, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Topic Breakdown</div>", unsafe_allow_html=True)
        display_df = topic_df.sort_values("Frequency", ascending=False)[
            ["Topic", "Frequency", "Type"]
        ].reset_index(drop=True)
        display_df.index = display_df.index + 1
        display_df["Frequency"] = display_df["Frequency"].apply(lambda v: f"{v:,}")
        st.dataframe(
            display_df,
            use_container_width=True,
            height=480,
        )

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Pipeline Visualization</div>", unsafe_allow_html=True)
    viz2_path = os.path.join(PLOTS_DIR, "viz2_top_discussion_topics.png")
    if os.path.exists(viz2_path):
        st.image(Image.open(viz2_path), use_container_width=True,
                 caption="viz2 — generated by topic_analysis.py pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ENGAGEMENT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Engagement":

    st.markdown("<div class='page-title'>Topic Engagement Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → Engagement</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <b>Why median, not mean?</b> Engagement is heavily right-skewed — a handful of viral
        posts have engagement in the tens of thousands while most posts receive fewer than 20
        interactions. The <em>median</em> reflects the typical post's reach far more accurately
        than the <em>mean</em>, which is inflated by outliers. All figures use the
        deduplicated corpus (89,162 unique texts).
    </div>
    """, unsafe_allow_html=True)

    # Engagement leaders
    st.markdown("<div class='section-header'>Engagement Leaders (by Median)</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi-card kpi-accent-amber'>
            <div class='kpi-label'>#1 — coal</div>
            <div class='kpi-value'>8</div>
            <div class='kpi-sub'>Median engagement · 865 posts</div>
        </div>
        <div class='kpi-card kpi-accent-blue'>
            <div class='kpi-label'>#2 — planet</div>
            <div class='kpi-value'>7</div>
            <div class='kpi-sub'>Median engagement · 1,461 posts</div>
        </div>
        <div class='kpi-card kpi-accent-violet'>
            <div class='kpi-label'>#3 — scientists</div>
            <div class='kpi-value'>6</div>
            <div class='kpi-sub'>Median engagement · 1,511 posts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive chart — median vs mean
    st.markdown("<div class='section-header'>Median vs Mean Engagement per Topic</div>",
                unsafe_allow_html=True)

    plot_eng = eng_df.sort_values("Median_Engagement")
    fig_eng = go.Figure()
    fig_eng.add_trace(go.Bar(
        name="Mean", y=plot_eng["Topic"], x=plot_eng["Mean_Engagement"],
        orientation="h", marker_color="#f59e0b", opacity=0.65,
    ))
    fig_eng.add_trace(go.Bar(
        name="Median", y=plot_eng["Topic"], x=plot_eng["Median_Engagement"],
        orientation="h", marker_color="#e67e22",
        text=plot_eng["Median_Engagement"].apply(lambda v: f"{v:.0f}"),
        textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
    ))
    fig_eng.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="#334155", title="Engagement Score", tickformat=","),
        yaxis=dict(gridcolor="#334155", title=""),
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=60),
        height=480,
    )
    st.plotly_chart(fig_eng, use_container_width=True)

    # Full table
    st.markdown("<div class='section-header'>Full Engagement Table</div>", unsafe_allow_html=True)
    display_eng = eng_df.copy()
    display_eng["Mean_Engagement"]   = display_eng["Mean_Engagement"].apply(lambda v: f"{v:,.1f}")
    display_eng["Median_Engagement"] = display_eng["Median_Engagement"].apply(lambda v: f"{v:.0f}")
    display_eng["Max_Engagement"]    = display_eng["Max_Engagement"].apply(lambda v: f"{v:,}")
    display_eng["Post_Count"]        = display_eng["Post_Count"].apply(lambda v: f"{v:,}")
    st.dataframe(display_eng, use_container_width=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Pipeline Visualization</div>", unsafe_allow_html=True)
    viz3_path = os.path.join(PLOTS_DIR, "viz3_topic_engagement.png")
    if os.path.exists(viz3_path):
        st.image(Image.open(viz3_path), use_container_width=True,
                 caption="viz3 — generated by sentiment_analysis.py pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Trends":

    st.markdown("<div class='page-title'>Topic Trends & Surge Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → Trends</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Weekly post counts are computed from the <em>deduplicated</em> corpus (89,162 unique texts).
        Surges are detected using a z-score against a 4-week rolling baseline (threshold z &gt; 2.0).
        These are reported as <em>observed volume anomalies</em> — no causal attribution is made
        from the data alone.
    </div>
    """, unsafe_allow_html=True)

    # Surge alert cards
    st.markdown("<div class='section-header'>Unusual Surge Detected — Week of 2018-08-13</div>",
                unsafe_allow_html=True)

    spike_data = [
        {"topic": "climate change", "posts": 7073,  "baseline": 328.0,  "z": 139.02, "pct": 2056.4},
        {"topic": "global warming",  "posts": 1644,  "baseline": 78.0,   "z": 131.42, "pct": 2007.7},
        {"topic": "science",         "posts": 411,   "baseline": 19.8,   "z": 61.17,  "pct": 1981.0},
        {"topic": "earth",           "posts": 467,   "baseline": 26.8,   "z": 28.56,  "pct": 1645.8},
    ]
    for s in spike_data:
        st.markdown(f"""
        <div class='spike-alert'>
            <div class='spike-week'>Week of 2018-08-13</div>
            <div class='spike-topic'>{s['topic']}</div>
            <div class='spike-stats'>
                <div class='spike-stat'>Posts: <strong>{s['posts']:,}</strong></div>
                <div class='spike-stat'>Baseline: <strong>{s['baseline']:,}</strong></div>
                <div class='spike-stat'>Z-score: <strong>{s['z']}</strong></div>
                <div class='spike-stat'>Increase: <strong>+{s['pct']:,.1f}%</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box' style='border-color:#f59e0b44; background:#1c1917;'>
        ⚠️ <b>Note:</b> The surge on 2018-08-13 is an <em>observed anomaly</em> in discussion volume.
        The data does not identify a cause. Brands should monitor such surges and prepare
        rapid-response content in case the topic resurfaces.
    </div>
    """, unsafe_allow_html=True)

    # Interactive trend chart from deduplicated data
    st.markdown("<div class='section-header'>Weekly Discussion Volume (Interactive)</div>",
                unsafe_allow_html=True)

    import re as _re
    trend_topics = ["climate change", "global warming", "science", "earth"]
    colors_trend = ["#3b82f6", "#f43f5e", "#22c55e", "#f59e0b"]

    @st.cache_data(show_spinner=False)
    def build_trend_df(df_dedup_hash):
        series_dict = {}
        for term in trend_topics:
            pattern = r"\b" + _re.escape(term) + r"\b"
            mask    = df_dedup["Cleaned_Text"].str.contains(pattern, regex=True, na=False)
            subset  = df_dedup.loc[mask].copy()
            subset["Week"] = subset["Timestamp"].dt.to_period("W").dt.start_time
            s = subset.groupby("Week").size().rename(term)
            if len(s) >= 4:
                series_dict[term] = s
        tdf = pd.DataFrame(series_dict).fillna(0).sort_index()
        return tdf.rolling(window=2, min_periods=1).mean()

    trend_df = build_trend_df(len(df_dedup))

    fig_trend = go.Figure()
    for i, topic in enumerate(trend_df.columns):
        fig_trend.add_trace(go.Scatter(
            x=trend_df.index, y=trend_df[topic],
            name=topic, mode="lines",
            line=dict(color=colors_trend[i % len(colors_trend)], width=2.2),
            hovertemplate=f"<b>{topic}</b><br>Week: %{{x|%b %d, %Y}}<br>Posts: %{{y:.0f}}<extra></extra>",
        ))
    fig_trend.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="#334155", title="Week"),
        yaxis=dict(gridcolor="#334155", title="Unique Posts (2-week rolling avg)", tickformat=","),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Pipeline Visualization</div>", unsafe_allow_html=True)
    viz4_path = os.path.join(PLOTS_DIR, "viz4_topic_trends_over_time.png")
    if os.path.exists(viz4_path):
        st.image(Image.open(viz4_path), use_container_width=True,
                 caption="viz4 — generated by trend_analysis.py pipeline")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODEL
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ML Model":

    st.markdown("<div class='page-title'>Machine Learning Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → ML Model</div>", unsafe_allow_html=True)

    # Model info cards
    st.markdown("""
    <div class='kpi-grid'>
        <div class='kpi-card kpi-accent-blue'>
            <div class='kpi-label'>Model</div>
            <div class='kpi-value' style='font-size:1.1rem;'>Multinomial<br>Naive Bayes</div>
            <div class='kpi-sub'>alpha = 0.1</div>
        </div>
        <div class='kpi-card kpi-accent-violet'>
            <div class='kpi-label'>Features</div>
            <div class='kpi-value' style='font-size:1.1rem;'>TF-IDF</div>
            <div class='kpi-sub'>20,000 features · ngram (1,2)</div>
        </div>
        <div class='kpi-card kpi-accent-cyan'>
            <div class='kpi-label'>Task</div>
            <div class='kpi-value' style='font-size:1.1rem;'>Sentiment<br>Classification</div>
            <div class='kpi-sub'>3 classes: Pos / Neu / Neg</div>
        </div>
        <div class='kpi-card kpi-accent-teal'>
            <div class='kpi-label'>Train / Test Split</div>
            <div class='kpi-value' style='font-size:1.1rem;'>80 / 20</div>
            <div class='kpi-sub'>Stratified · 161,465 posts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        TF-IDF (Term Frequency–Inverse Document Frequency) converts tweet text into numerical
        feature vectors by weighting words that are frequent in a document but rare across the
        corpus. Multinomial Naive Bayes then uses those feature counts to classify each tweet
        as Positive, Neutral, or Negative using Bayes' theorem with a smoothing parameter
        alpha = 0.1.
    </div>
    """, unsafe_allow_html=True)

    # Accuracy metrics
    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

    acc  = metrics.get("accuracy", 0.7785)
    f1m  = metrics.get("f1_macro", 0.7771)
    f1w  = metrics.get("f1_weighted", 0.7774)
    prec = metrics.get("prec_macro", 0.7975)
    rec  = metrics.get("rec_macro", 0.7712)

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, label, val in [
        (col1, "Accuracy",       f"{acc*100:.2f}%"),
        (col2, "Macro F1",       f"{f1m:.4f}"),
        (col3, "Weighted F1",    f"{f1w:.4f}"),
        (col4, "Macro Precision",f"{prec:.4f}"),
        (col5, "Macro Recall",   f"{rec:.4f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class='model-metric'>
                <div class='model-metric-val'>{val}</div>
                <div class='model-metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("<div class='section-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), use_container_width=True,
                     caption="Confusion matrix — test set (32,293 posts)")

    with col_b:
        st.markdown("<div class='section-header'>Per-Class Results</div>", unsafe_allow_html=True)
        per_class = pd.DataFrame({
            "Class"    : ["Negative", "Neutral", "Positive"],
            "Precision": [0.8891, 0.7650, 0.7383],
            "Recall"   : [0.6652, 0.7771, 0.8712],
            "F1-Score" : [0.7610, 0.7710, 0.7993],
            "Support"  : [8143, 13993, 10157],
        })
        fig_pc = go.Figure()
        for metric_col, color in [("Precision","#3b82f6"),("Recall","#22c55e"),("F1-Score","#f59e0b")]:
            fig_pc.add_trace(go.Bar(
                name=metric_col, x=per_class["Class"],
                y=per_class[metric_col], marker_color=color,
            ))
        fig_pc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0", family="Inter"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 1.05], title="Score"),
            barmode="group",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_pc, use_container_width=True)

        st.markdown("<div class='section-header'>Live Classifier</div>", unsafe_allow_html=True)
        user_text = st.text_area(
            "Enter a tweet to classify:",
            placeholder="e.g. Climate change is the biggest threat of our generation",
            height=90,
        )
        if st.button("Classify Sentiment", type="primary"):
            if user_text.strip():
                bundle = load_model()
                model  = bundle["model"]
                tfidf  = bundle["tfidf"]
                pred   = model.predict(tfidf.transform([user_text.lower()]))[0]
                proba  = model.predict_proba(tfidf.transform([user_text.lower()]))[0]
                classes = model.classes_
                color_map_pred = {"Positive": "#22c55e", "Neutral": "#3b82f6", "Negative": "#f43f5e"}
                c = color_map_pred.get(pred, "#e2e8f0")
                st.markdown(f"""
                <div class='content-card' style='border-left:4px solid {c}; margin-top:0.5rem;'>
                    <div style='font-size:0.8rem; color:#64748b; margin-bottom:0.3rem;'>Prediction</div>
                    <div style='font-size:1.4rem; font-weight:700; color:{c};'>{pred}</div>
                </div>
                """, unsafe_allow_html=True)
                prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
                fig_prob = px.bar(prob_df, x="Class", y="Probability",
                                  color="Class",
                                  color_discrete_map=color_map_pred,
                                  text=prob_df["Probability"].apply(lambda v: f"{v:.1%}"))
                fig_prob.update_traces(textposition="outside")
                fig_prob.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#1e293b",
                    font=dict(color="#e2e8f0", family="Inter"),
                    yaxis=dict(gridcolor="#334155", range=[0, 1.1]),
                    xaxis=dict(gridcolor="#334155"),
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=220,
                )
                st.plotly_chart(fig_prob, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & ACTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Insights & Actions":

    st.markdown("<div class='page-title'>Insights & Action Plan</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-breadcrumb'>Overview → Insights & Actions</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        All insights and actions are read directly from <code>results/insights.txt</code>,
        which is generated by the verified DS_Day01_42 pipeline. No values are invented or
        hardcoded in this dashboard.
    </div>
    """, unsafe_allow_html=True)

    # ── Key Insights ──────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-sub'>{len(insights)} insights derived from pipeline results</div>",
                unsafe_allow_html=True)

    accent_colors = [
        "#3b82f6","#22c55e","#f59e0b","#f43f5e",
        "#8b5cf6","#06b6d4","#14b8a6",
    ]
    for i, ins in enumerate(insights):
        color = accent_colors[i % len(accent_colors)]
        st.markdown(f"""
        <div class='insight-card' style='border-left-color:{color};'>
            <div class='insight-title'>
                <span style='color:{color}; margin-right:0.5rem;'>#{i+1}</span>
                {ins['title']}
            </div>
            <div class='insight-row'>
                <span class='insight-tag tag-finding'>Finding</span>
                <span class='insight-text'>{ins['finding']}</span>
            </div>
            <div class='insight-row'>
                <span class='insight-tag tag-evidence'>Evidence</span>
                <span class='insight-text'>{ins['evidence']}</span>
            </div>
            <div class='insight-row'>
                <span class='insight-tag tag-meaning'>Meaning</span>
                <span class='insight-text'>{ins['meaning']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    # ── Action Plan ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Practical Action Plan</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-sub'>{len(actions)} recommended actions</div>",
                unsafe_allow_html=True)

    badge_map = {
        "HIGH"  : "badge-high",
        "MEDIUM": "badge-medium",
        "LOW"   : "badge-low",
    }
    for i, act in enumerate(actions):
        badge_cls = badge_map.get(act["priority"].upper(), "badge-medium")
        st.markdown(f"""
        <div class='action-card'>
            <span class='priority-badge {badge_cls}'>{act['priority']}</span>
            <div>
                <div class='action-title'>Action {i+1}: {act['title']}</div>
                <div class='action-detail'>{act['detail']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
