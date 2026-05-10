# Restaurant Sentiment Analysis

**Hospitality & Service | Natural Language Processing | Binary Classification**

A machine learning project that automatically classifies restaurant reviews as positive or negative — with a business-first focus on **not missing negative feedback**.

---

## The Business Problem

A restaurant owner receives hundreds of reviews. The real risk is not reading a bad one. A missed negative review means an unaddressed complaint, a lost customer, and a pattern that never gets fixed.

This project frames sentiment classification as a **Negative Recall problem** — the model is optimised to catch as many negative reviews as possible, even if that means occasionally flagging a positive one as suspicious.

> A false alarm costs a manager 10 seconds. A missed complaint can cost a customer permanently.

---

## Results at a Glance

| Model | Test Neg Recall | Test F1 | Train-Test Gap |
|---|---|---|---|
| **XGBoost** ← winner | **0.96** | 0.51 | **0.018** |
| Random Forest | 0.92 | 0.67 | 0.055 |
| Logistic Regression | 0.82 | 0.67 | 0.052 |

**XGBoost catches 96 out of every 100 negative reviews** with virtually zero overfitting (gap = 0.018).

---

## Project Structure

```
sentiment_analysis.ipynb    ← Main notebook (Google Colab)
Restaurant_Reviews.tsv      ← Dataset (1,000 reviews, tab-separated)
README.md                   ← This file
TECHNICAL_ANALYSIS.md       ← Full methodology and modelling decisions
EXECUTIVE_SUMMARY.md        ← Business-focused summary and recommendations
```

---

## Pipeline Overview

```
Raw Reviews
    ↓
Text Cleaning         (URLs, punctuation, lowercase, numbers)
    ↓
Lemmatization         (POS-aware, WordNetLemmatizer)
    ↓
Custom Stopwords      (remove words shared equally across both classes)
    ↓
TF-IDF Vectorisation  (unigrams + bigrams, sublinear TF, inside Pipeline)
    ↓
Classification        (LR / XGBoost / Random Forest)
    ↓
Evaluation            (Neg Recall, F1, Train-Test Gap, 5-Fold CV)
```

---

## Key Design Decisions

**Why Negative Recall, not Accuracy or F1?**
For a restaurant owner, all misclassification errors are not equal. Missing a negative review (False Negative for the negative class) is far more costly than a false alarm. Negative Recall directly measures the percentage of actual negative reviews the model catches.

**Why XGBoost wins?**
`learning_rate=0.01` makes the ensemble extremely conservative — it needs very strong positive evidence before committing to a positive prediction. This bias directly translates to high Negative Recall. Combined with `max_depth=3` and `n_estimators=200`, the model generalises cleanly with a train-test gap of just 0.018.

**Why sklearn Pipelines?**
Each model wraps TF-IDF + classifier in a single Pipeline object. This prevents data leakage (the vectoriser only sees training data), simplifies cross-validation, and mirrors how the model would work in production.

**Why different feature counts per model?**
- LR (100 features): linear model with no internal feature selection — too many features on 800 samples causes memorisation
- XGBoost (300): tree splits filter uninformative features internally at each node
- RF (200): same principle as XGBoost, slightly fewer features further boosts Negative Recall

---

## Dataset

- **Source:** UCI Machine Learning Repository — Restaurant Reviews
- **Size:** 1,000 reviews → 996 after dropping 4 duplicates
- **Classes:** 499 Positive (50.1%) / 497 Negative (49.9%) — near-perfectly balanced
- **Format:** Tab-separated (`.tsv`), two columns: `Review` (text) and `Liked` (0/1)
- **Split:** 80% train (796) / 20% test (200), stratified

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Language |
| pandas, numpy | Data handling |
| scikit-learn | Pipelines, TF-IDF, LR, RF, metrics, CV |
| XGBoost | Gradient boosting classifier |
| NLTK | Tokenisation, POS tagging, lemmatization |
| matplotlib, seaborn | Visualisation |
| WordCloud | Word cloud generation |
| Google Colab | Notebook environment |

---

## How to Run

1. Upload `Restaurant_Reviews.tsv` to your Google Drive at `MyDrive/Restaurant_Reviews.tsv`
2. Open `sentiment_analysis.ipynb` in Google Colab
3. Run **Runtime → Run all**

All libraries are installed in the first cell (`!pip install nltk wordcloud xgboost`).

---

## What the Notebook Covers

1. **Data Inspection** — missing values, class balance, sample reviews
2. **EDA** — review length distributions, top keywords per class, word clouds, bigram analysis
3. **Text Preprocessing** — cleaning, POS-aware lemmatization, custom stopword removal
4. **Modelling** — Logistic Regression, XGBoost, Random Forest inside sklearn Pipelines
5. **Model Comparison** — sorted by Negative Recall with train-test gap analysis
6. **Model Design Choices** — why these models, why not BERT or GridSearchCV
7. **Prediction Probabilities** — confidence scores and distributions per model
8. **Cross-Validation** — 5-fold CV confirming results are stable
9. **Error Analysis** — what fools the model and why, with concrete preprocessing fixes
10. **Business Recommendations** — actionable insights tied directly to complaint patterns

---

## Limitations & Next Steps

- **996 reviews is a small dataset** — more data would allow larger vocabularies without overfitting
- **Bag-of-words cannot handle negation** — *"not bad"* and *"never disappointing"* confuse TF-IDF models
- **Subtle negative words are underweighted** — *"mediocre"*, *"uninspired"* appear rarely; adding them to a custom negative vocabulary would reduce false positives
- **Next step:** With 10,000+ reviews, fine-tuning a pre-trained BERT model would handle context and negation natively
