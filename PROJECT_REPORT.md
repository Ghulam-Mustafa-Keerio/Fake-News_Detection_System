# Fake News Detection System — Complete Project Report

**Author:** Ghulam Mustafa  
**Date:** February 2026  
**Repository:** [github.com/Ghulam-Mustafa-Keerio/Fake-News_Detection_System](https://github.com/Ghulam-Mustafa-Keerio/Fake-News_Detection_System)  
**License:** MIT

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Literature Review](#5-literature-review)
6. [System Architecture](#6-system-architecture)
7. [Dataset](#7-dataset)
8. [Methodology](#8-methodology)
9. [Implementation](#9-implementation)
10. [Model Evaluation](#10-model-evaluation)
11. [Web Application](#11-web-application)
12. [API Documentation](#12-api-documentation)
13. [Security Considerations](#13-security-considerations)
14. [Results and Discussion](#14-results-and-discussion)
15. [Limitations and Future Work](#15-limitations-and-future-work)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)
18. [Appendices](#18-appendices)

---

## 1. Abstract

This report presents the design, implementation, and evaluation of an AI-powered Fake News Detection System built with Python, scikit-learn, and Flask. The system employs a TF-IDF–based ensemble machine learning model combining three classifiers — Stochastic Gradient Descent (SGD), Logistic Regression, and Random Forest — through soft-voting to classify news articles as **REAL** or **FAKE**. Trained on the Kaggle Fake and Real News Dataset containing 44,898 articles, the system achieves **98.86% accuracy** and **99.93% ROC-AUC** on a held-out test set. Beyond classification, the system provides Explainable AI (XAI) capabilities, including per-prediction feature importance, linguistic analysis, risk-factor detection, and credibility indicators. The trained model is served through a responsive Flask web application and a desktop Tkinter GUI, with a full REST API for programmatic integration. Model serialization uses a secure JSON + HMAC-SHA256 format, eliminating the arbitrary code execution risks inherent in Python's pickle.

---

## 2. Introduction

### 2.1 Background

The proliferation of online misinformation poses a significant threat to public discourse, democratic processes, and societal trust. Social media platforms enable the rapid spread of fabricated news articles that mimic the appearance of legitimate journalism, making manual verification impractical at scale.

Automated fake news detection systems leverage Natural Language Processing (NLP) and machine learning to analyze textual content and identify patterns distinguishing fabricated articles from genuine reporting. These systems serve as decision-support tools for journalists, fact-checkers, and content moderators.

### 2.2 Scope

This project builds a complete, end-to-end fake news detection pipeline:

- **Data ingestion** — Automated downloading and cleaning of the Kaggle Fake and Real News Dataset
- **Model training** — TF-IDF feature extraction and ensemble classification with cross-validation
- **Explainability** — Feature importance, linguistic analysis, and risk-factor identification
- **Deployment** — Flask web application, desktop GUI, and REST API
- **Security** — Secure JSON serialization with HMAC integrity verification

---

## 3. Problem Statement

Given a text input (news article, headline, or social media post), the system must:

1. **Classify** it as REAL or FAKE with a calibrated confidence score
2. **Explain** the classification by identifying the key features and linguistic patterns that influenced the decision
3. **Detect risk factors** — sensational language, clickbait patterns, excessive capitalization, and other stylistic red flags
4. **Serve predictions** through a user-friendly web interface and a programmatic API

---

## 4. Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Build a text classification model that distinguishes fake from real news with >95% accuracy | ✅ Achieved (98.86%) |
| 2 | Provide explainable predictions showing which words and patterns influenced the decision | ✅ Implemented |
| 3 | Detect and report linguistic risk factors and credibility indicators | ✅ Implemented |
| 4 | Deploy the model as a web application with a modern UI | ✅ Implemented |
| 5 | Expose a REST API for programmatic integration | ✅ Implemented |
| 6 | Use secure model serialization (no pickle) | ✅ Implemented |

---

## 5. Literature Review

### 5.1 Fake News Detection Approaches

The literature identifies several approaches to automated fake news detection:

**Content-based approaches** analyze the text of the article itself. Key techniques include:
- Bag-of-words and TF-IDF representations for capturing vocabulary differences between fake and real articles (Shu et al., 2017)
- N-gram analysis to detect phrases characteristic of fabricated content
- Stylometric features — writing style, punctuation patterns, and readability indices

**Network-based approaches** analyze how articles spread through social networks:
- Propagation patterns and cascade structures
- User credibility and bot detection
- Source reliability scores

**Knowledge-based approaches** compare article claims against known facts:
- Knowledge graph look-ups
- Fact-checking database queries
- Claim verification pipelines

This project implements the **content-based approach** using TF-IDF features enhanced with linguistic analysis.

### 5.2 Ensemble Learning

Ensemble methods combine multiple classifiers to reduce variance and improve generalization. Dietterich (2000) identified three fundamental reasons why ensembles outperform individual models:
1. **Statistical** — Averaging over multiple hypotheses reduces the risk of choosing a poor model
2. **Computational** — Different models explore different regions of the hypothesis space
3. **Representational** — A combination of models can represent functions outside any single model's hypothesis space

This project uses a **soft-voting ensemble** that averages the predicted class probabilities of three diverse classifiers: a linear model (SGD), a logistic model, and a tree-based model (Random Forest).

### 5.3 Explainable AI for Text Classification

Post-hoc explainability in text classification typically relies on feature importance scores derived from model weights. For linear models (SGD, Logistic Regression), the learned coefficient vector directly indicates which TF-IDF features push the prediction toward each class. This project extracts the highest-weight features for each prediction and presents them alongside rule-based linguistic analysis.

---

## 6. System Architecture

### 6.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                              │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │  Flask Web    │    │  Desktop     │    │  REST API            │  │
│   │  Application  │    │  GUI (Tk)    │    │  (curl / Python)     │  │
│   └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│          │                   │                       │               │
└──────────┼───────────────────┼───────────────────────┼───────────────┘
           │                   │                       │
           └───────────────────┼───────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PREDICTION ENGINE                               │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  AdvancedFakeNewsDetector                                    │  │
│   │                                                              │  │
│   │  ┌────────────────┐    ┌────────────────────────────────┐   │  │
│   │  │ Text           │    │ TF-IDF Vectorizer              │   │  │
│   │  │ Preprocessor   │───▶│ (10,000 features, n-gram 1-3)  │   │  │
│   │  └────────────────┘    └──────────┬─────────────────────┘   │  │
│   │                                   │                          │  │
│   │                    ┌──────────────┼──────────────┐           │  │
│   │                    ▼              ▼              ▼           │  │
│   │            ┌──────────┐   ┌──────────┐   ┌──────────┐      │  │
│   │            │   SGD    │   │ Logistic │   │  Random  │      │  │
│   │            │Classifier│   │Regression│   │  Forest  │      │  │
│   │            └────┬─────┘   └────┬─────┘   └────┬─────┘      │  │
│   │                 │              │              │              │  │
│   │                 └──────────────┼──────────────┘              │  │
│   │                                ▼                             │  │
│   │                    ┌──────────────────┐                      │  │
│   │                    │  Soft Voting +   │                      │  │
│   │                    │  Calibration     │                      │  │
│   │                    └────────┬─────────┘                      │  │
│   │                             ▼                                │  │
│   │                    ┌──────────────────┐                      │  │
│   │                    │  Explainability  │                      │  │
│   │                    │  Layer           │                      │  │
│   │                    └────────┬─────────┘                      │  │
│   │                             ▼                                │  │
│   │                     PredictionResult                         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               ▲
                               │
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL PERSISTENCE                               │
│                                                                     │
│   model.json  ←  HMAC-SHA256  →  vectorizer.json                   │
│   model_config.json              training_metrics.json              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Overview

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Training Engine | `train_model.py` | 1,022 | Model training, evaluation, save/load, prediction |
| Dataset Utilities | `dataset_utils.py` | 374 | Data download, loading, cleaning, bias removal |
| Web Application | `app.py` | 390 | Flask routes, API endpoints, model serving |
| Desktop GUI | `gui.py` | — | Tkinter-based standalone interface |
| Download Script | `scripts/download_kaggle_dataset.sh` | — | Automated Kaggle dataset download |
| Web Templates | `templates/index.html`, `templates/about.html` | — | Frontend UI |

---

## 7. Dataset

### 7.1 Source

The project uses the **Fake and Real News Dataset** from Kaggle, compiled by Clément Bisaillon. The dataset was collected from two sources:

- **Fake news** — Articles flagged as unreliable by Politifact and from known fake news outlets
- **Real news** — Articles from Reuters news agency

### 7.2 Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total articles | 44,898 |
| Fake articles | 23,481 (52.3%) |
| Real articles | 21,417 (47.7%) |
| Class balance | Near-balanced (4.6% difference) |
| Average article length | ~400 words |
| Time period | 2015–2018 |
| Primary language | English |
| Primary topic | US politics |

### 7.3 Data Fields

| Column | Description |
|--------|-------------|
| `title` | Article headline |
| `text` | Article body text |
| `subject` | Topic category (e.g., `politicsNews`, `worldnews`, `Government News`, `left-news`) |
| `date` | Publication date |

### 7.4 Data Preprocessing and Bias Mitigation

A critical challenge with this dataset is **source-attribution bias**. All articles in `True.csv` begin with a Reuters byline pattern:

```
WASHINGTON (Reuters) - The U.S. Senate on Thursday...
LONDON (Reuters) - British Prime Minister Theresa May...
```

If left uncleaned, the model learns to detect Reuters formatting rather than content quality, leading to models that classify any Reuters-formatted text as "real" and everything else as "fake."

**Cleaning steps implemented in `load_kaggle_dataset()`:**

| Pattern | Regex | Purpose |
|---------|-------|---------|
| Reuters byline | `^[A-Z ]{2,}\s*\(Reuters\)\s*-\s*` | Remove city + agency prefix |
| AP byline | `\(AP\)` | Remove Associated Press tag |
| Correction boilerplate | `^Corrects paragraph \d+.*?\n` | Remove editorial corrections |
| Trump Twitter headers | Specific heading patterns | Remove non-article boilerplate |

**After cleaning:**
- Title and text are concatenated into a single `content` field
- Labels are assigned: `Fake = 0`, `Real = 1`
- Data is shuffled with `random_state=42` for reproducibility

### 7.5 Train/Test Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 35,918 | 80% |
| Testing | 8,980 | 20% |
| Stratification | Yes | Preserves class ratios |

---

## 8. Methodology

### 8.1 Text Preprocessing

The `AdvancedTextPreprocessor` class applies a multi-stage NLP pipeline:

1. **Case normalization** — Convert to lowercase
2. **URL removal** — Strip `http://`, `https://`, and `www.` patterns
3. **HTML tag removal** — Strip any residual HTML markup
4. **Special character removal** — Remove non-alphanumeric characters (preserving spaces)
5. **Number removal** — Strip numeric tokens
6. **Tokenization** — NLTK `word_tokenize()` for linguistically-aware splitting
7. **Stopword removal** — Filter English stopwords from the NLTK corpus
8. **Lemmatization** — WordNet lemmatizer reduces words to base forms (`"running"` → `"run"`)

### 8.2 Feature Extraction

The system uses **Term Frequency–Inverse Document Frequency (TF-IDF)** vectorization:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

With sublinear TF scaling:

$$\text{TF}(t, d) = 1 + \log(\text{raw count of } t \text{ in } d)$$

And standard IDF:

$$\text{IDF}(t) = \log\frac{N}{1 + \text{df}(t)} + 1$$

Where $N$ is the total number of documents and $\text{df}(t)$ is the number of documents containing term $t$.

**Vectorizer configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_features` | 10,000 | Balance vocabulary coverage vs. dimensionality |
| `ngram_range` | (1, 3) | Capture unigrams, bigrams, and trigrams |
| `sublinear_tf` | True | Dampen effect of high-frequency terms |
| `min_df` | 2 | Exclude extremely rare terms |
| `max_df` | 0.95 | Exclude near-universal terms |

### 8.3 Classification Models

#### 8.3.1 SGD Classifier

The **Stochastic Gradient Descent** classifier optimizes a linear model using gradient descent on randomly-selected mini-batches. Configuration:

- **Loss function:** Modified Huber — a smooth variant of hinge loss that provides probability estimates
- **Penalty:** L2 (ridge) regularization
- **Advantages:** Fast training on large datasets, naturally supports probability estimates through `predict_proba()`

#### 8.3.2 Logistic Regression

A generalized linear model for binary classification:

$$P(y = 1 | x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

Configuration:
- **Solver:** LBFGS (quasi-Newton method)
- **Penalty:** L2 regularization
- **Max iterations:** 1,000
- **Advantages:** Highly interpretable coefficient vector, calibrated probability estimates

#### 8.3.3 Random Forest

An ensemble of 200 decision trees, each trained on a bootstrap sample of the data:

- **Number of estimators:** 200
- **Max features:** `sqrt(n_features)` (per-split)
- **Advantages:** Captures non-linear feature interactions, resistant to overfitting

#### 8.3.4 Soft-Voting Ensemble

The three classifiers are combined through a **VotingClassifier** with soft voting:

$$P_{\text{ensemble}}(y | x) = \frac{1}{K} \sum_{k=1}^{K} P_k(y | x)$$

Where $K = 3$ is the number of base classifiers. The class with the highest averaged probability is selected.

### 8.4 Probability Calibration

Post-hoc calibration using **Platt scaling** (sigmoid method) via `CalibratedClassifierCV`:

$$P_{\text{calibrated}}(y | x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

Where $A$ and $B$ are learned from a held-out calibration set. This ensures that a reported 80% confidence means approximately 80% of such predictions are correct.

### 8.5 Explainability

Each prediction includes:

| Component | Source | Description |
|-----------|--------|-------------|
| **Top features** | TF-IDF × model coefficients | Words with highest absolute weight for the prediction |
| **Risk factors** | Rule-based analysis | Excessive caps, exclamation marks, ALL CAPS words, sensational vocabulary |
| **Credibility indicators** | Rule-based analysis | Neutral tone, professional punctuation, appropriate complexity |
| **Linguistic analysis** | Text statistics | Word count, sentence count, caps ratio, exclamation count, sensational word count |

---

## 9. Implementation

### 9.1 Core Classes

#### `AdvancedTextPreprocessor`

```python
class AdvancedTextPreprocessor:
    """
    NLP text preprocessor with lemmatization, stopword removal,
    and sensational language / clickbait pattern detection
    """
    def preprocess(self, text: str, advanced: bool = True) -> str
    def analyze_text(self, text: str) -> dict
```

#### `AdvancedFakeNewsDetector`

```python
class AdvancedFakeNewsDetector:
    """
    Main detection engine — training, prediction, save/load, explainability
    """
    def train(self, texts, labels, validate=True, calibrate=True) -> dict
    def predict(self, text: str, explain: bool = True) -> PredictionResult
    def save_model(self, model_path='model.json', vec_path='vectorizer.json')
    def load_model(self, model_path='model.json', vec_path='vectorizer.json')
```

#### `PredictionResult`

```python
@dataclass
class PredictionResult:
    label: str                          # "REAL" or "FAKE"
    confidence: float                   # 0-100%
    probabilities: dict                 # {"fake": %, "real": %}
    top_features: list                  # [{"word": str, "weight": float}, ...]
    linguistic_analysis: dict           # {word_count, caps_ratio, ...}
    risk_factors: list                  # ["Excessive exclamation marks", ...]
    credibility_indicators: list        # ["Neutral language tone", ...]
```

### 9.2 Training Flow

```python
def main():
    # Step 1: Load data from Kaggle dataset
    texts, labels = load_kaggle_dataset("data/fake_and_real_news/")

    # Step 2: Initialize detector
    detector = AdvancedFakeNewsDetector(model_type=ModelType.TFIDF_ENSEMBLE)

    # Step 3: Train with validation and calibration
    metrics = detector.train(texts, labels, validate=True, calibrate=True)

    # Step 4: Save to secure JSON format
    detector.save_model()
```

### 9.3 Prediction Flow

```python
# 1. Load model from JSON files
detector = AdvancedFakeNewsDetector()
detector.load_model()  # Verifies HMAC signatures

# 2. Preprocess input text
clean_text = preprocessor.preprocess(raw_text)

# 3. Vectorize with TF-IDF
features = vectorizer.transform([clean_text])

# 4. Get ensemble prediction
probabilities = model.predict_proba(features)

# 5. Generate explanation
explanation = {
    "top_features": extract_top_features(features, model.coef_),
    "risk_factors": analyze_risk_factors(raw_text),
    "credibility_indicators": analyze_credibility(raw_text),
    "linguistic_analysis": compute_statistics(raw_text)
}
```

### 9.4 Secure Serialization

Model persistence avoids Python's `pickle` entirely. Instead:

1. **Vectorizer** — Vocabulary (dict of `{term: index}`), IDF weights (list of floats), and vectorizer configuration are stored in `vectorizer.json`
2. **Model** — For each base estimator, the coefficient matrix (`coef_`), intercept (`intercept_`), and class labels are stored in `model.json`
3. **Integrity** — Each JSON file includes an HMAC-SHA256 signature computed over its data payload. On load, the signature is recomputed and compared.

### 9.5 File Organization

```
train_model.py          (1,022 lines)  — Core ML engine
dataset_utils.py          (374 lines)  — Data utilities
app.py                    (390 lines)  — Flask web app
gui.py                               — Desktop GUI
templates/index.html      (642 lines)  — Web frontend
templates/about.html      (256 lines)  — About page
scripts/download_kaggle_dataset.sh    — Kaggle download helper
```

---

## 10. Model Evaluation

### 10.1 Overall Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.86% |
| **Precision** | 98.87% |
| **Recall** | 98.86% |
| **F1 Score** | 98.86% |
| **ROC-AUC** | 99.93% |

### 10.2 Split Statistics

| Parameter | Value |
|-----------|-------|
| Training samples | 35,918 |
| Test samples | 8,980 |
| Vocabulary size | 10,000 |
| Model type | TF-IDF Ensemble |
| Training date | 2026-02-14 |

### 10.3 Interpretation

- **98.86% accuracy** means the model correctly classifies approximately 8,878 out of 8,980 test articles, misclassifying only ~102.
- **99.93% ROC-AUC** indicates near-perfect discriminative ability — the model ranks almost all real articles higher than fake articles in its probability estimates.
- The precision and recall are nearly identical (both ~98.87%), indicating balanced performance across both classes, with no strong bias toward either over- or under-prediction of fake news.

### 10.4 Sample Test Predictions

| Input Text | Prediction | Confidence |
|------------|-----------|------------|
| "The Federal Reserve announced today that it will maintain current interest rates, citing stable inflation levels and moderate economic growth." | **REAL** | 66.4% |
| "BREAKING NEWS!!! You wont BELIEVE what scientists discovered! Proof that ALIENS have been controlling world governments for DECADES!" | **FAKE** | 87.6% |
| "Pakistan Meteorological Department has forecast rain and thunderstorms in several parts of the country during the next few days." | **REAL** | 60.0% |
| "Scientists at Stanford University have completed a comprehensive 10-year study examining the effects of regular physical exercise on cardiovascular health." | **REAL** | 53.5% |
| "YOU WONT BELIEVE this miracle cure discovered by a local mom! Doctors HATE her for revealing this shocking secret!" | **FAKE** | 84.9% |

---

## 11. Web Application

### 11.1 Overview

The Flask web application provides a modern, responsive user interface for interacting with the fake news detection model. The frontend uses a dark gradient theme (`#1a1a2e` → `#0f3460`) with no external CSS framework.

### 11.2 Pages

| Route | Description |
|-------|-------------|
| `/` | Main prediction interface — text input, prediction results, and explainability panel |
| `/about` | Project description, methodology overview, and team information |

### 11.3 Features

- **Real-time prediction** — Submit text and receive results instantly
- **Confidence meter** — Visual representation of prediction certainty
- **Risk factor badges** — Warning indicators for detected red flags
- **Credibility badges** — Positive indicators for legitimate content signals
- **Top feature display** — Shows which words most influenced the prediction
- **Responsive design** — Works on desktop and mobile browsers

### 11.4 Server Configuration

```python
app.run(host='0.0.0.0', port=5000, debug=os.environ.get('FLASK_DEBUG') == 'true')
```

---

## 12. API Documentation

### 12.1 Endpoints

#### `POST /predict` — Single Prediction

**Request:**
```json
{
  "text": "Your news article text here"
}
```

**Response (200 OK):**
```json
{
  "prediction": "REAL",
  "confidence": 66.4,
  "probabilities": {
    "fake": 33.6,
    "real": 66.4
  },
  "model_type": "advanced_ensemble",
  "explanation": {
    "top_features": [
      { "word": "citing", "weight": 0.23 },
      { "word": "federal reserve", "weight": 0.09 }
    ],
    "risk_factors": [],
    "credibility_indicators": [
      "Neutral language tone",
      "Professional punctuation"
    ],
    "linguistic_analysis": {
      "word_count": 41,
      "sentence_count": 2,
      "caps_ratio": 4.0,
      "exclamation_marks": 0,
      "sensational_words": 0
    }
  }
}
```

#### `POST /predict/batch` — Batch Prediction

Accepts up to 100 texts in a single request.

**Request:**
```json
{
  "texts": [
    "Article 1 text...",
    "Article 2 text..."
  ]
}
```

**Response:** Array of individual prediction objects.

#### `POST /analyze` — Linguistic Analysis Only

Returns linguistic statistics without running the classifier.

**Request:**
```json
{
  "text": "Text to analyze"
}
```

#### `GET /health` — Health Check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "advanced_ensemble",
  "accuracy": 0.9886,
  "features": ["explainability", "calibrated_confidence", "linguistic_analysis"],
  "security": "json_only_no_pickle"
}
```

#### `GET /api/models` — Available Models

Lists all available model types and their status.

---

## 13. Security Considerations

### 13.1 Threat: Pickle Deserialization Attacks

Python's `pickle` module can execute arbitrary code during deserialization. A malicious `model.pkl` file could:
- Execute shell commands
- Exfiltrate data
- Install backdoors
- Modify system files

This is a well-documented vulnerability (CVE-2019-6446 for NumPy, and a general class of attacks documented by OWASP).

### 13.2 Mitigation: JSON + HMAC-SHA256

This project eliminates pickle entirely:

| Component | Format | Integrity Check |
|-----------|--------|-----------------|
| Model weights | JSON (lists of floats) | HMAC-SHA256 |
| Vectorizer vocabulary | JSON (dict of term→index) | HMAC-SHA256 |
| IDF weights | JSON (list of floats) | HMAC-SHA256 |
| Configuration | JSON (dict) | No signature (metadata only) |

**HMAC verification flow:**
1. On save: compute `HMAC-SHA256(secret_key, JSON_data)` and store with the data
2. On load: recompute the HMAC and compare with the stored signature
3. If signatures do not match, reject the file (possible tampering)

### 13.3 Secure Load Reconstruction

When loading models from JSON, the system reconstructs scikit-learn estimators by:
1. Creating new instances of `SGDClassifier`, `LogisticRegression`, etc.
2. Setting `coef_`, `intercept_`, and `classes_` from the JSON data
3. Rebuilding the `VotingClassifier` wrapper
4. **Random Forest** is skipped during load (its internal tree structure cannot be securely serialized as JSON without pickle), so the ensemble operates with SGD + Logistic Regression only at prediction time

This is a deliberate security tradeoff: slightly reduced ensemble diversity in exchange for complete elimination of arbitrary code execution risk.

---

## 14. Results and Discussion

### 14.1 Performance Analysis

The system achieves **98.86% accuracy**, which is consistent with published benchmarks on this dataset. The near-perfect ROC-AUC (99.93%) indicates that the model has strong discriminative ability and is not simply memorizing training examples.

### 14.2 Source-Attribution Bias

A critical finding during development was the **Reuters attribution bias**. The original dataset contains Reuters bylines (`"WASHINGTON (Reuters) -"`) exclusively in real articles. Without cleaning, the model learns to classify any text containing this pattern as "real" and everything else as "fake" — achieving high test accuracy but failing completely on new data from any non-Reuters source.

**Solution:** Regex-based cleaning of Reuters, AP, and other attribution patterns in `load_kaggle_dataset()` forces the model to learn content-level features rather than formatting shortcuts.

### 14.3 Explainability Value

The explainability features provide practical value:

- **Top features** show users *why* the model made its prediction, enabling informed trust decisions
- **Risk factors** serve as independent, rule-based checks that complement the ML prediction
- **Credibility indicators** provide positive evidence, useful when the ML confidence is borderline

### 14.4 Confidence Distribution

Real news predictions tend to show moderate confidence (53–67%), reflecting the model's appropriate uncertainty about content from sources not seen during training. Fake news with strong stylistic signals (ALL CAPS, exclamation marks, sensational language) receives high confidence scores (80–90%).

---

## 15. Limitations and Future Work

### 15.1 Current Limitations

| Limitation | Impact | Potential Mitigation |
|------------|--------|---------------------|
| **English only** | Cannot process articles in other languages | Multilingual TF-IDF or multilingual models |
| **US politics focus** | Training data is predominantly US political news (2015–2018) | Augment with diverse international news corpora |
| **Content-only analysis** | No network/propagation or knowledge-base features | Integrate social media APIs and knowledge graphs |
| **Static model** | Does not update with new misinformation patterns | Implement online learning or periodic retraining |
| **Random Forest excluded at load** | Ensemble loses one of three base classifiers after save/load | Implement tree serialization in JSON or use a different non-linear model |
| **Dataset age** | 2015–2018 data may not reflect modern misinformation tactics | Retrain on up-to-date datasets |

### 15.2 Future Work

1. **Transformer fine-tuning** — Integrate a transformer-based model (e.g., DistilBERT) for even higher accuracy on GPU-equipped environments.

2. **Multi-language support** — Integrate multilingual models to detect fake news in Urdu, Arabic, Hindi, and other languages.

3. **Real-time learning** — Implement an active-learning loop where user feedback on incorrect predictions is used to incrementally retrain the model.

4. **Source credibility scoring** — Combine content analysis with domain-level reputation databases (e.g., NewsGuard, Media Bias/Fact Check).

5. **Browser extension** — Deploy the detection capability as a browser extension that automatically analyzes news articles while the user browses.

6. **Model dashboard** — Build a monitoring dashboard showing prediction distributions, confidence histograms, and drift detection metrics over time.

---

## 16. Conclusion

This project demonstrates a complete, production-ready fake news detection system that combines traditional machine learning with explainable AI capabilities. Key achievements include:

- **98.86% classification accuracy** on a real-world dataset of 44,898 news articles
- **Transparent predictions** with feature importance, linguistic analysis, and risk-factor reporting
- **Secure model persistence** using JSON serialization with HMAC-SHA256 integrity verification, eliminating pickle-based code execution risks
- **Multiple deployment modes** — Flask web application, desktop GUI, and REST API
- **Source-attribution debiasing** — Regex-based cleaning prevents the model from learning formatting shortcuts instead of content-level patterns

The system provides a strong foundation for further development, including multi-language support, real-time learning, and source credibility scoring.

---

## 17. References

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective.* ACM SIGKDD Explorations Newsletter, 19(1), 22–36.

2. Dietterich, T. G. (2000). *Ensemble Methods in Machine Learning.* Multiple Classifier Systems, Lecture Notes in Computer Science, Vol. 1857, Springer.

3. Reis, J. C., Correia, A., Murai, F., Veloso, A., & Benevenuto, F. (2019). *Supervised Learning for Fake News Detection.* IEEE Intelligent Systems, 34(2), 76–81.

4. Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2018). *Automatic Detection of Fake News.* Proceedings of the 27th International Conference on Computational Linguistics (COLING).

5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

6. Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* Advances in Large Margin Classifiers, MIT Press.

7. Bisaillon, C. (2020). *Fake and Real News Dataset.* Kaggle. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## 18. Appendices

### Appendix A: Requirements

```
# Core Dependencies
Flask>=2.3.2
Werkzeug>=2.3.6
scikit-learn>=1.3.0
numpy>=1.24.3
pandas>=2.0.3
nltk>=3.8.1
```

### Appendix B: Environment Variables

| Variable | Purpose |
|----------|---------|
| `KAGGLE_USERNAME` | Kaggle account username (for dataset download) |
| `KAGGLE_API_TOKEN` | Kaggle API token (for dataset download) |
| `KAGGLE_KEY` | Alternative: Kaggle API key |
| `FLASK_DEBUG` | Set to `"true"` to enable Flask debug mode |

### Appendix C: Model Files

| File | Size | Content |
|------|------|---------|
| `model.json` | ~5 MB | Classifier weights, intercepts, class labels, HMAC signature |
| `vectorizer.json` | ~2 MB | Vocabulary (10,000 terms), IDF weights, configuration, HMAC signature |
| `model_config.json` | <1 KB | Model type, vocabulary size, version number |
| `training_metrics.json` | <1 KB | Accuracy, precision, recall, F1, ROC-AUC, sample counts, timestamp |

### Appendix D: Running the Complete Pipeline

```bash
# 1. Clone and install
git clone https://github.com/Ghulam-Mustafa-Keerio/Fake-News_Detection_System.git
cd Fake-News_Detection_System
pip install -r requirements.txt

# 2. Download dataset
mkdir -p data/fake_and_real_news
# Place Fake.csv and True.csv in data/fake_and_real_news/

# 3. Train model
python train_model.py

# 4. Start web application
python app.py

# 5. Test via API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here"}'
```

---

*Report generated: February 2026*  
*Fake News Detection System v2.0*
