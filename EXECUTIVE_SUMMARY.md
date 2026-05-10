# Executive Summary — Restaurant Sentiment Analysis

---

## The Problem

Every day, restaurant owners receive reviews they never read. When a negative review goes unnoticed, a complaint goes unanswered, a recurring problem goes unfixed, and a customer never comes back.

This project builds a machine learning system that **automatically reads and classifies restaurant reviews** — identifying which ones are negative so that managers can act on them immediately.

The key design decision: the system is optimised **not to miss negative reviews**, even if that means occasionally flagging a positive one. A false alarm takes a manager ten seconds to dismiss. A missed complaint can cost a customer for life.

---

## What Was Built

A natural language processing (NLP) pipeline that:

1. **Cleans raw review text** — removes noise, standardises language, and reduces words to their root forms
2. **Identifies discriminative vocabulary** — removes words that appear equally in both positive and negative reviews (zero signal), keeping only words that actually distinguish one from the other
3. **Converts text to numbers** — using TF-IDF (Term Frequency-Inverse Document Frequency), which weights words by how uniquely they signal a sentiment class
4. **Classifies reviews** — three machine learning models were trained and compared; the best was selected based on the business objective

---

## The Result

| Model | Negative Reviews Caught | Notes |
|---|---|---|
| **XGBoost** ← recommended | **96 out of 100** | Tightest generalisation; lowest overfitting |
| Random Forest | 92 out of 100 | Strong alternative |
| Logistic Regression | 82 out of 100 | Interpretable baseline |

**XGBoost is the recommended model.** Across a sample of 200 test reviews, it correctly identified 96% of actual negative reviews — missing only 4%. It also generalises cleanly: trained and test performance are nearly identical (gap = 0.018), confirming the model has not memorised training data.

---

## What the Model Found in the Data

### Patterns in Negative Reviews

The most common negative signals — words and phrases that reliably predict a complaint:

| Signal | What It Means |
|---|---|
| *"slow"*, *"took long"*, *"wait"* | Service speed is the #1 operational complaint |
| *"cold"* | Food temperature — a food runner or kitchen timing issue |
| *"never go back"* | **Churn intent** — the highest-risk signal in any review |
| *"rude"*, *"unfriendly"* | Staff attitude — a training and culture issue |
| *"bad"*, *"never"*, *"dont"*, *"much"* | Core negative sentiment vocabulary |

### Patterns in Positive Reviews

The most common positive signals — what satisfied customers praise:

| Signal | What It Means |
|---|---|
| *"great"*, *"amazing"*, *"delicious"* | Food quality — the primary driver of positive sentiment |
| *"nice"*, *"friendly"* | Atmosphere and staff — loyalty drivers |
| *"great place"*, *"great food"* | Holistic satisfaction — customers who are likely to return |

---

## Business Recommendations

| Operational Finding | Recommended Action |
|---|---|
| Service speed dominates negative reviews (*"slow"*, *"took long"*, *"wait"*) | Review kitchen workflows; set ticket time targets; add staff at peak hours |
| *"cold"* appears frequently in complaints | Upgrade food runner system; use heated plates; review timing between kitchen and table |
| *"never go back"* is a churn signal | Flag these reviews immediately for manager follow-up; consider a service recovery protocol (direct outreach, voucher) |
| *"rude"*, *"unfriendly"* recur in negative bigrams | Invest in hospitality training; add customer service KPIs to staff performance reviews |
| *"great"*, *"delicious"*, *"friendly"* drive positive reviews | Feature the most-praised dishes in marketing materials; publicly recognise high-performing staff |

---

## How It Works in Practice

Once deployed, the model can be integrated into a review management dashboard:

1. A new review arrives (from Google, Yelp, TripAdvisor, etc.)
2. The model classifies it: **Positive** or **Negative**
3. Negative reviews are flagged and surfaced to a manager within minutes
4. The manager responds, resolves, and tracks the pattern

At XGBoost's 0.96 Negative Recall, only **4 in every 100 negative reviews** would slip through undetected — compared to a manual reading process where reviews may go unread for days.

---

## Limitations

| Limitation | Impact | Fix |
|---|---|---|
| Small dataset (996 reviews) | Model vocabulary is limited; rare words are underweighted | Collect more reviews; retrain periodically |
| Bag-of-words cannot handle negation | *"not bad"* and *"never disappointing"* may be misclassified | Upgrade to BERT with 10,000+ reviews |
| Subtle negative words missed (*"mediocre"*, *"uninspired"*) | Some well-worded complaints slip through | Add to custom negative vocabulary in preprocessing |
| Short positive phrases missed (*"seat immediately"*, *"quick order"*) | Some brief positive reviews flagged as negative | Add to custom positive vocabulary; adjust short-review handling |

---

## Recommended Next Steps

1. **Deploy XGBoost** as the production classifier — Neg Recall = 0.96 is production-ready for a restaurant use case
2. **Expand the negative vocabulary** — add *"mediocre"*, *"uninspired"*, *"verge"* to preprocessing to reduce the 4% miss rate further
3. **Retrain quarterly** — as menus, staff, and language evolve, the model should be updated with fresh reviews
4. **Scale to 10,000+ reviews** — at that volume, fine-tuning a pre-trained BERT language model would eliminate the bag-of-words limitations entirely, handling negation and context natively

---

## Summary

This project demonstrates that even a relatively small dataset (996 reviews) — when approached with the right business objective and the right model choices — can produce a highly effective complaint detection system. XGBoost's 0.96 Negative Recall means a restaurant owner using this model would catch nearly every negative review automatically, turning reactive complaint management into a proactive, data-driven operation.
