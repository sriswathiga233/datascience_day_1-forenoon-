"""
naive_bayes_model.py
--------------------
Trains a Multinomial Naive Bayes classifier to predict Sentiment_Label
from cleaned tweet text using a TF-IDF pipeline.

Key design decisions:
  - Stratified 80/20 train/test split preserves class proportions.
  - TF-IDF is fitted ONLY on training data to prevent data leakage.
  - Sparse matrices used throughout for memory efficiency.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import joblib


def run_naive_bayes(df: pd.DataFrame,
                    models_dir: str,
                    plots_dir: str,
                    results_dir: str) -> dict:
    """
    Trains and evaluates the Naive Bayes model.
    Saves the model, confusion matrix plot, and text results.
    Returns a dict of evaluation metrics.
    """
    print("\n" + "=" * 60)
    print("STEP 9 & 10 - NAIVE BAYES MODEL & EVALUATION")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    X = df["Cleaned_Text"].fillna("")
    y = df["Sentiment_Label"]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTraining samples : {len(X_train):,}")
    print(f"Test samples     : {len(X_test):,}")
    print(f"Class distribution in test set:\n{y_test.value_counts().to_string()}")

    # TF-IDF fitted on TRAINING data only (prevents data leakage)
    tfidf = TfidfVectorizer(
        max_features=20000,   # cap features for efficiency
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,    # log(1+tf) reduces impact of very frequent terms
        min_df=3,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)   # fit + transform on train only
    X_test_tfidf  = tfidf.transform(X_test)         # transform only on test

    print(f"\nTF-IDF vocabulary size : {len(tfidf.vocabulary_):,}")
    print(f"Feature matrix (train) : {X_train_tfidf.shape}")

    # Train Multinomial Naive Bayes
    model = MultinomialNB(alpha=0.1)   # alpha=0.1: mild Laplace smoothing
    model.fit(X_train_tfidf, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, digits=4)
    cm       = confusion_matrix(
        y_test, y_pred, labels=["Positive", "Neutral", "Negative"]
    )

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro"
    )
    prec_wtd, rec_wtd, f1_wtd, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print(f"\nAccuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"\nClassification Report:\n{report}")
    print(f"Macro    - Precision: {prec_macro:.4f}  Recall: {rec_macro:.4f}  F1: {f1_macro:.4f}")
    print(f"Weighted - Precision: {prec_wtd:.4f}  Recall: {rec_wtd:.4f}  F1: {f1_wtd:.4f}")

    # Metric explanations
    print("""
Metric Explanations (in context of sentiment prediction):
  Accuracy   : % of tweets whose sentiment was correctly predicted overall.
  Precision  : Of all tweets predicted as a given sentiment, how many were correct.
               High precision = fewer false alarms.
  Recall     : Of all tweets that truly belong to a sentiment, how many were found.
               High recall = fewer missed cases.
  F1-score   : Harmonic mean of precision and recall.
  Macro avg  : Average across classes equally (important for imbalanced data).
  Weighted   : Average weighted by class size (reflects real-world distribution).
""")

    # Save confusion matrix plot
    labels = ["Positive", "Neutral", "Negative"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_title("Naive Bayes - Confusion Matrix\n(Sentiment Classification)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("[Saved] confusion_matrix.png")

    # Save model
    joblib.dump(
        {"model": model, "tfidf": tfidf},
        os.path.join(models_dir, "naive_bayes_model.pkl")
    )
    print("[Saved] naive_bayes_model.pkl")

    # Save text results
    result_text = (
        "NAIVE BAYES MODEL RESULTS\n"
        + "=" * 50 + "\n"
        + f"Training samples : {len(X_train):,}\n"
        + f"Test samples     : {len(X_test):,}\n"
        + f"TF-IDF features  : {len(tfidf.vocabulary_):,}\n\n"
        + f"Accuracy         : {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        + f"Classification Report:\n{report}\n"
        + f"Macro    - Precision: {prec_macro:.4f}  Recall: {rec_macro:.4f}  F1: {f1_macro:.4f}\n"
        + f"Weighted - Precision: {prec_wtd:.4f}  Recall: {rec_wtd:.4f}  F1: {f1_wtd:.4f}\n"
    )
    with open(os.path.join(results_dir, "model_results.txt"), "w", encoding="utf-8") as f:
        f.write(result_text)
    print("[Saved] model_results.txt")

    return {
        "accuracy"   : accuracy,
        "report"     : report,
        "cm"         : cm,
        "f1_macro"   : f1_macro,
        "f1_weighted": f1_wtd,
        "prec_macro" : prec_macro,
        "rec_macro"  : rec_macro,
    }
