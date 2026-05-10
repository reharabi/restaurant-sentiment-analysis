# Technical Analysis — Restaurant Sentiment Analysis

---

## 1. Problem Framing

This is a **binary text classification** task: given a restaurant review, predict whether it is Positive (Liked=1) or Negative (Liked=0).

The standard approach would optimise for accuracy or macro F1. However, the business context changes the objective. A restaurant owner's cost function is **asymmetric**:

| Error Type | Description | Business Cost |
|---|---|---|
| False Negative (for class 0) | Predicted Positive, actually Negative | **High** — complaint goes unaddressed, customer lost |
| False Positive (for class 0) | Predicted Negative, actually Positive | **Low** — manager spends 10 seconds on a non-issue |

The correct metric is therefore **Negative class Recall** (`recall_score(..., pos_label=0)`): the proportion of actual negative reviews that the model correctly identifies as negative.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository — Restaurant Reviews |
| Raw size | 1,000 reviews |
| After deduplication | 996 reviews (4 duplicates dropped) |
| Positive class (Liked=1) | 499 (50.1%) |
| Negative class (Liked=0) | 497 (49.9%) |
| Train split | 796 reviews (80%) |
| Test split | 200 reviews (20%) |
| Split strategy | Stratified — 50/50 balance preserved in both sets |

The near-perfect class balance means no oversampling, class weighting, or imbalance correction is needed. Accuracy and F1 are aligned and both reliable as secondary metrics.

---

## 3. Text Preprocessing Pipeline

All preprocessing is applied before any model fitting to prevent leakage.

### 3.1 Text Cleaning

```python
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'#', '', text)                 # remove hashtag symbol
    text = text.lower()                            # lowercase
    text = re.sub(r'\d+', '', text)               # remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()     # normalise whitespace
```

### 3.2 POS-Aware Lemmatization

Standard stemming (e.g. Porter Stemmer) applies blunt suffix-chopping rules: *"studies"* → *"studi"*, *"caring"* → *"car"*. This loses real word form.

POS-aware lemmatization maps each token to its correct dictionary base form using its grammatical role:

```python
def get_wordnet_pos(tag):
    if tag.startswith('J'):  return 'a'   # adjective
    elif tag.startswith('V'): return 'v'  # verb
    elif tag.startswith('N'): return 'n'  # noun
    elif tag.startswith('R'): return 'r'  # adverb
    else:                     return 'n'
```

Examples:
- *"loved"* → *"love"* (verb)
- *"running"* → *"run"* (verb)
- *"orders"* → *"order"* (noun)
- *"better"* → *"good"* (adjective — this is the key win over stemming)

For short restaurant reviews (median ~10 words), every word matters. Accurate root forms improve TF-IDF weight estimation.

### 3.3 Custom Stopword Detection

Standard NLTK stopwords remove function words (*"the"*, *"is"*). However, domain-specific words like *"food"* and *"place"* appear at high frequency in **both** positive and negative reviews — they carry zero discriminative signal.

We identify these data-driven custom stopwords by finding the intersection of the top-15 words across both classes:

```python
pos_top15 = set([w for w, _ in Counter(pos_token_list).most_common(15)])
neg_top15 = set([w for w, _ in Counter(neg_token_list).most_common(15)])
custom_stop_words = pos_top15.intersection(neg_top15)
```

These are removed from all token lists before TF-IDF vectorisation.

---

## 4. Feature Engineering

### TF-IDF Vectorisation

Each model uses TF-IDF with the following shared settings:

```python
TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # log(1 + tf) instead of raw tf
    max_features=N        # model-specific (see below)
)
```

**Why bigrams?** Single words are ambiguous — *"good"* alone is positive, but *"not good"* and *"never good"* are clearly negative. Including bigrams lets TF-IDF capture two-word sentiment phrases and partial negation patterns.

**Why sublinear_tf?** A word appearing 10 times does not carry 10x the signal of a word appearing once. Log-scaling dampens the effect of high-frequency terms.

**Why different max_features per model?**

| Model | max_features | Reasoning |
|---|---|---|
| Logistic Regression | 100 | Linear model — no internal feature selection. With 800 training samples, restricting to 100 features prevents memorisation of rare training-specific words |
| XGBoost | 300 | Tree splits filter uninformative features at each node — can handle more features without memorising them |
| Random Forest | 200 | Same principle as XGBoost; slightly fewer features pushes the model to focus on the most consistent negative signals |

### DenseTransformer

XGBoost and Random Forest require dense numpy arrays. TF-IDF produces sparse matrices. A minimal custom transformer handles the conversion inside the Pipeline:

```python
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.toarray() if issparse(X) else X
```

---

## 5. Models

All models are wrapped in sklearn Pipelines: `TF-IDF → [DenseTransformer →] Classifier`. The Pipeline ensures the vectoriser is fitted only on training data — preventing leakage during cross-validation.

### 5.1 Logistic Regression (Baseline)

```python
Pipeline([
    ('vec', TfidfVectorizer(max_features=100, ngram_range=(1,2), sublinear_tf=True)),
    ('mod', LogisticRegression(max_iter=500, C=0.1, random_state=42))
])
```

| Metric | Value |
|---|---|
| Test Neg Recall | 0.82 |
| Test F1 | 0.67 |
| Train-Test Gap | ~0.052 |

**C=0.1** applies strong L2 regularisation — the inverse of regularisation strength, so smaller C = stronger penalty. This prevents any single feature (word) from dominating predictions.

LR is the interpretable baseline: its coefficients directly show which words push predictions toward positive or negative, making it useful for explaining model behaviour to non-technical stakeholders.

### 5.2 XGBoost (Winner)

```python
Pipeline([
    ('vec',   TfidfVectorizer(max_features=300, ngram_range=(1,2), sublinear_tf=True)),
    ('dense', DenseTransformer()),
    ('mod',   xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.01, max_depth=3,
        eval_metric='logloss', random_state=42, verbosity=0
    ))
])
```

| Metric | Value |
|---|---|
| Test Neg Recall | **0.96** |
| Test F1 | 0.51 |
| Train-Test Gap | **0.018** |

**Why XGBoost achieves the highest Negative Recall:**

`learning_rate=0.01` means each of the 200 boosting rounds contributes only 1% of its correction to the ensemble. After 200 rounds the model has built up a conservative aggregate that requires very strong positive evidence before committing to a positive prediction. Any borderline review defaults toward Negative — directly maximising Negative Recall.

`max_depth=3` keeps individual trees as shallow weak learners. Each tree partitions the feature space into at most 8 regions — too coarse to memorise individual training samples.

The trade-off: **Positive Recall = 0.36** — many positive reviews are flagged as negative (false alarms). This is the acceptable cost of a 0.96 Negative Recall for the restaurant use case.

### 5.3 Random Forest

```python
Pipeline([
    ('vec',   TfidfVectorizer(max_features=200, ngram_range=(1,2), sublinear_tf=True)),
    ('dense', DenseTransformer()),
    ('mod',   RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1))
])
```

| Metric | Value |
|---|---|
| Test Neg Recall | 0.92 |
| Test F1 | 0.67 |
| Train-Test Gap | ~0.055 |

Unlike XGBoost (sequential boosting), Random Forest is a **bagging** ensemble — 100 trees are trained independently on different bootstrap samples of the training data and random feature subsets at each split. Predictions are averaged across all trees.

`max_depth=4` constrains each tree to at most 16 leaf regions — shallow enough to generalise without overfitting on 800 samples.

---

## 6. Model Comparison

| Model | Train F1 | Test F1 | Train Neg Recall | Test Neg Recall | Gap |
|---|---|---|---|---|---|
| **XGBoost** | ~0.53 | 0.51 | ~0.97 | **0.96** | **0.018** |
| Random Forest | ~0.72 | 0.67 | ~0.97 | 0.92 | 0.055 |
| Logistic Regression | ~0.72 | 0.67 | ~0.88 | 0.82 | 0.052 |

Sorted by Test Neg Recall — the primary business metric.

**Key observations:**
- XGBoost dominates on both Negative Recall (0.96) and generalisation (gap=0.018)
- LR and RF have similar Test F1 (~0.67) but RF has a meaningfully higher Neg Recall (0.92 vs 0.82)
- All three models have acceptable gaps — none are memorising training data

---

## 7. Cross-Validation

5-fold cross-validation is run on the training set for all three models using the same final hyperparameters:

```python
cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
```

CV validates that test set results are not the product of a lucky train/test split. If CV mean F1 ≈ test F1, the split is representative.

---

## 8. Error Analysis

**Model evaluated:** XGBoost (best Neg Recall)
**Misclassification rate:** 68/200 (34%)

This rate is expected and acceptable — XGBoost is optimised for Negative Recall, not overall accuracy. The 34% misclassification is almost entirely False Positives (positive reviews flagged as negative), not False Negatives (negative reviews missed).

### False Positives — model predicted Positive, actually Negative

| Review (preprocessed) | Root cause |
|---|---|
| *"selection best"* | Word *"best"* is a strong positive TF-IDF signal — model latches onto it |
| *"bean rice mediocre best"* | *"best"* overrides *"mediocre"*; *"mediocre"* is rare in training data |
| *"main also uninspired"* | *"uninspired"* is genuinely negative but appears too rarely to carry TF-IDF weight |

**Fix:** Add *"mediocre"*, *"uninspired"*, *"verge"* to a custom negative seed vocabulary — force TF-IDF to treat them as negative anchors regardless of frequency.

### False Negatives — model predicted Negative, actually Positive

| Review (preprocessed) | Root cause |
|---|---|
| *"awesome selection beer"* | *"beer"* frequently appears in negative reviews (*"beer was flat"*, etc.) — association overpowers *"awesome"* |
| *"seat immediately"* | Very short, no strong positive keywords — model defaults to negative in absence of evidence |
| *"quick even order"* | Same issue — positive intent expressed through operational efficiency words, not sentiment words |

**Fix:** Reinforce *"awesome"*, *"immediately"*, *"quick"* as positive signals in preprocessing. Consider adding a minimum-length filter or a short-review handling rule.

### Root cause — bag-of-words limitation

TF-IDF treats every word independently. It has no understanding of:
- **Negation:** *"not bad"* is processed as separate tokens *"not"* + *"bad"*
- **Context:** *"mediocre at best"* contains *"best"* — a positive word in isolation
- **Sarcasm / irony:** Cannot be detected without sequence modelling

BERT-based models handle all three natively through bidirectional attention across the full sentence.

---

## 9. Why Not GridSearchCV?

GridSearchCV automates hyperparameter search but obscures the reasoning behind what works. The key insight in this project — that `learning_rate=0.01` → conservatism → high Negative Recall — was found by reasoning about the objective, not brute-force grid search. Manual tuning makes the mechanism transparent and reproducible.

## 10. Why Not BERT?

| Consideration | Detail |
|---|---|
| Dataset size | 800 training samples — far below the 10,000+ typically needed to fine-tune BERT without overfitting |
| Compute | BERT fine-tuning requires GPU; this project runs on free Colab |
| Explainability | TF-IDF coefficients (LR) and feature importances (XGBoost) are directly interpretable; BERT attention weights are not |
| Upgrade path | With 10,000+ reviews, `bert-base-uncased` or a domain-specific `restaurant-bert` fine-tune would be the correct next step |
