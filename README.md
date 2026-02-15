# 🔍 Fake News Detection System

An AI-powered machine learning system for detecting fake news articles using Natural Language Processing (NLP) and Explainable AI. The system combines a TF-IDF–based ensemble classifier with linguistic analysis and risk-factor detection to deliver transparent, interpretable predictions. Built with Python, Flask, and scikit-learn.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey.svg)](https://flask.palletsprojects.com/)

---

## Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Training Pipeline](#-training-pipeline)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Usage Examples](#-usage-examples)
- [Security](#-security)
- [Technologies Used](#-technologies-used)
- [Research Background](#-research-background)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Ensemble ML Model** | Soft-voting ensemble of SGD, Logistic Regression, and Random Forest for robust predictions |
| **Explainable AI** | Shows the top TF-IDF features that influenced each prediction |
| **Linguistic Analysis** | Detects sensational language, clickbait patterns, excessive caps, and exclamation marks |
| **Risk Factor Detection** | Identifies red flags common in fake news (ALL CAPS words, sensational vocabulary) |
| **Credibility Indicators** | Highlights positive signals — neutral tone, professional punctuation, appropriate complexity |
| **Confidence Calibration** | Sigmoid-calibrated probabilities for reliable confidence scores |
| **Secure Serialization** | JSON + HMAC-SHA256 signatures — no pickle files, no arbitrary code execution risk |
| **Web Interface** | Modern, responsive Flask-based UI with dark gradient theme |
| **Desktop GUI** | Tkinter-based standalone desktop application |
| **REST API** | Full API with single prediction, batch prediction, analysis, and health-check endpoints |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Ghulam-Mustafa-Keerio/Fake-News_Detection_System.git
cd Fake-News_Detection_System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

```bash
# Option A: Using the provided helper script (requires Kaggle CLI)
pip install --user kaggle
export KAGGLE_USERNAME=your_username
export KAGGLE_API_TOKEN=your_token
bash scripts/download_kaggle_dataset.sh .

# Option B: Manual download
# Download from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# Place Fake.csv and True.csv into data/fake_and_real_news/
mkdir -p data/fake_and_real_news
mv Fake.csv True.csv data/fake_and_real_news/
```

### 4. Train the Model

```bash
python train_model.py
```

### 5. Run the Web Application

```bash
python app.py
# Open http://localhost:5000 in your browser
```

---

## 📥 Dataset

This project uses the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon.

| Metric | Value |
|--------|-------|
| **Total Articles** | 44,898 |
| **Fake Articles** | 23,481 |
| **Real Articles** | 21,417 |
| **Source (Real)** | Reuters news agency |
| **Topics** | Politics, world news, government, economics |

### Data Preprocessing

The dataset loading pipeline (`dataset_utils.py`) applies the following cleaning steps to prevent source-attribution bias:

- **Reuters pattern removal** — Strips `"CITY (Reuters) -"` prefixes from Real articles so the model learns content quality rather than source formatting
- **AP pattern removal** — Strips `"(AP)"` attributions  
- **Correction-note removal** — Removes `"Corrects paragraph X…"` boilerplate
- **Title + text concatenation** — Merges headline and body into a single feature for richer context
- **Shuffling** — Randomized with `random_state=42` for reproducibility

### Dataset Directory Layout

```
data/
└── fake_and_real_news/
    ├── Fake.csv    # 23,481 fake news articles
    └── True.csv    # 21,417 real news articles (Reuters)
```

---

## 📁 Project Structure

```
Fake-News_Detection_System/
│
├── train_model.py              # Core ML training with TF-IDF ensemble & secure save/load
├── app.py                      # Flask web application (routes & API)
├── gui.py                      # Desktop GUI application (Tkinter)
├── dataset_utils.py            # Dataset download, loading, and cleaning utilities
│
├── model.json                  # Trained ensemble model (secure JSON + HMAC)
├── vectorizer.json             # TF-IDF vectorizer (secure JSON + HMAC)
├── model_config.json           # Model metadata (type, vocab size, version)
├── training_metrics.json       # Training accuracy, precision, recall, F1, ROC-AUC
│
├── templates/
│   ├── index.html              # Main web interface (dark gradient theme)
│   └── about.html              # About page
│
├── scripts/
│   └── download_kaggle_dataset.sh   # Automated Kaggle dataset download
│
├── data/
│   └── fake_and_real_news/     # Kaggle CSV files (Fake.csv, True.csv)
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🧠 How It Works

### 1. Text Preprocessing Pipeline

```
Raw Text
  │
  ├── Lowercase conversion
  ├── URL removal (http/https/www patterns)
  ├── HTML tag stripping
  ├── Special character removal
  ├── Number removal
  ├── Tokenization (NLTK word_tokenize)
  ├── Stopword filtering (English)
  ├── Lemmatization (WordNet)
  │
  └── Clean Text
```

The `AdvancedTextPreprocessor` class also detects:
- **Sensational words** — `shocking`, `miracle`, `bombshell`, `conspiracy`, etc. (15 patterns)
- **Clickbait patterns** — `"you won't believe"`, `"doctors hate"`, `"what happened next"`, etc. (7 regex patterns)

### 2. Feature Extraction (TF-IDF)

| Parameter | Value |
|-----------|-------|
| Max features | 10,000 |
| N-gram range | (1, 3) — unigrams, bigrams, trigrams |
| TF scaling | Sublinear (`1 + log(tf)`) |
| Min document frequency | 2 |
| Max document frequency | 95% |

### 3. Ensemble Classification

```
                    ┌─────────────────────┐
                    │   TF-IDF Features    │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────────┐ ┌──────────┐ ┌────────────────┐
     │ SGD Classifier │ │ Logistic │ │ Random Forest  │
     │ (Modified      │ │Regression│ │ (200 trees)    │
     │  Huber Loss)   │ │          │ │                │
     └───────┬────────┘ └────┬─────┘ └───────┬────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │    Soft Voting       │
                    │  (Probability Avg)   │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Sigmoid Calibration  │
                    └─────────┬───────────┘
                              ▼
                    ┌─────────────────────┐
                    │  REAL / FAKE + %     │
                    └─────────────────────┘
```

### 4. Explainability Layer

Each prediction returns:

- **Top Features** — The TF-IDF terms with the highest positive/negative weight for the prediction
- **Risk Factors** — Detected red flags (e.g., "Excessive exclamation marks", "High capitalization ratio", "Multiple ALL CAPS words")
- **Credibility Indicators** — Positive signals (e.g., "Neutral language tone", "Professional punctuation", "Appropriate sentence complexity")
- **Linguistic Analysis** — Quantitative metrics: word count, sentence count, caps ratio, exclamation mark count, sensational word count

---

## 📊 Model Architecture

| Component | Configuration | Purpose |
|-----------|---------------|---------|
| **Vectorizer** | TF-IDF, 10,000 features, n-gram (1,3), sublinear TF | Convert text to numerical feature vectors |
| **SGD Classifier** | Modified Huber loss, L2 penalty | Fast linear classifier with probability estimates |
| **Logistic Regression** | L2 regularization, `lbfgs` solver | Interpretable linear model with calibrated probabilities |
| **Random Forest** | 200 estimators, max depth limited | Non-linear decision boundary, feature importance |
| **VotingClassifier** | Soft voting (averaged probabilities) | Combine all three classifiers for robustness |
| **Calibration** | `CalibratedClassifierCV`, sigmoid method | Produce well-calibrated probability scores |

---

## 🏋️ Training Pipeline

The training process in `train_model.py` follows this workflow:

1. **Data Loading** — Load 44,898 articles from the Kaggle dataset via `load_kaggle_dataset()`
2. **Text Cleaning** — Strip Reuters/AP attributions, remove boilerplate patterns
3. **Train/Test Split** — 80% training (35,918 samples), 20% testing (8,980 samples), stratified
4. **TF-IDF Vectorization** — Fit the vectorizer on training data; transform both sets
5. **Ensemble Training** — Train SGD, Logistic Regression, and Random Forest; combine via `VotingClassifier`
6. **Probability Calibration** — Apply `CalibratedClassifierCV` with sigmoid calibration
7. **Evaluation** — Compute accuracy, precision, recall, F1, ROC-AUC on the test set
8. **Secure Save** — Serialize to JSON with HMAC-SHA256 integrity signatures

### Run Training

```bash
python train_model.py
```

Expected output:
```
Step 1/4: Loading data...
  Loaded 44898 articles from Kaggle dataset
Step 2/4: Initializing detector...
Step 3/4: Training model...
  Training completed!
Step 4/4: Saving model (SECURE JSON format)...
  Model saved securely!
```

---

## 📈 Model Performance

Evaluation metrics on the held-out test set (8,980 articles):

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.86% |
| **Precision** | 98.87% |
| **Recall** | 98.86% |
| **F1 Score** | 98.86% |
| **ROC-AUC** | 99.93% |

| Split | Samples |
|-------|---------|
| Training set | 35,918 |
| Test set | 8,980 |
| Vocabulary size | 10,000 |

### Sample Predictions

**Fake News:**
```
Input:  "BREAKING NEWS!!! You wont BELIEVE what scientists discovered!
         Proof that ALIENS have been controlling world governments for DECADES!"
Output: FAKE (87.6% confidence)

Risk Factors:
  ⚠️ Excessive exclamation marks
  ⚠️ High capitalization ratio
  ⚠️ Multiple ALL CAPS words
```

**Real News:**
```
Input:  "The Federal Reserve announced today that it will maintain current
         interest rates, citing stable inflation levels and moderate economic growth."
Output: REAL (66.4% confidence)

Credibility Indicators:
  ✓ Appropriate sentence complexity
  ✓ Neutral language tone
  ✓ Professional punctuation
```

---

## 🌐 API Reference

The Flask application exposes the following REST endpoints:

### `GET /`
Serves the main web interface.

### `POST /predict`
Analyze a single text and return a prediction with explainability.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 66.4,
  "probabilities": { "fake": 33.6, "real": 66.4 },
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

### `POST /predict/batch`
Analyze up to 100 texts in a single request.

**Request:**
```json
{
  "texts": ["Article 1 text...", "Article 2 text..."]
}
```

### `POST /analyze`
Perform linguistic analysis only (no prediction).

**Request:**
```json
{ "text": "Text to analyze" }
```

### `GET /health`
Health check returning model status and capabilities.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "advanced_ensemble",
  "features": ["explainability", "calibrated_confidence", "linguistic_analysis"]
}
```

### `GET /api/models`
List all available model types.

---

## 👨‍💻 Usage Examples

### Python API

```python
from train_model import AdvancedFakeNewsDetector

# Load the trained model
detector = AdvancedFakeNewsDetector()
detector.load_model()

# Make a prediction with full explainability
result = detector.predict("Your news text here", explain=True)

print(f"Prediction:   {result.label}")
print(f"Confidence:   {result.confidence}%")
print(f"Fake prob:    {result.probabilities['fake']}%")
print(f"Real prob:    {result.probabilities['real']}%")
print(f"Top features: {result.top_features}")
print(f"Risk factors: {result.risk_factors}")
print(f"Credibility:  {result.credibility_indicators}")
```

### Web Application

```bash
python app.py
# Navigate to http://localhost:5000
```

### Desktop GUI

```bash
python gui.py
```

### Command Line Training

```bash
# Train on the Kaggle dataset
python train_model.py
```

---

## 🔒 Security

This project prioritizes secure model serialization over convenience.

### Why Not Pickle?

Pickle files can execute arbitrary Python code when loaded, creating a severe security vulnerability:

```python
# DANGER — Pickle can run malicious code
import pickle
pickle.loads(malicious_data)  # Executes attacker's code!
```

### Our Approach: JSON + HMAC-SHA256

| Risk | Mitigation |
|------|------------|
| **Arbitrary code execution** | Eliminated — JSON cannot execute code |
| **Model tampering** | HMAC-SHA256 signature verified on every load |
| **Opaque model files** | All files are human-readable JSON |
| **Supply-chain attacks** | No dependency on pickle or joblib for serialization |

```python
# SAFE — JSON cannot execute code
import json
model_data = json.load(open('model.json'))  # Just data, no code execution
# Signature is verified before use
```

**Model files produced:**
- `model.json` — Classifier weights and parameters + HMAC signature
- `vectorizer.json` — TF-IDF vocabulary and IDF weights + HMAC signature
- `model_config.json` — Model metadata (type, vocabulary size, version)
- `training_metrics.json` — Performance metrics from the last training run

---

## 🔧 Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.8+ |
| **Web Framework** | Flask 2.3+ |
| **Machine Learning** | scikit-learn 1.3+ |
| **NLP** | NLTK 3.8+ (tokenization, lemmatization, stopwords) |
| **Feature Extraction** | TF-IDF Vectorizer (10,000 features, n-gram 1–3) |
| **Ensemble Learning** | VotingClassifier (SGD + Logistic Regression + Random Forest) |
| **Serialization** | JSON + HMAC-SHA256 (secure, no pickle) |
| **Data Processing** | pandas 2.0+, NumPy 1.24+ |
| **Desktop GUI** | Tkinter |

---

## 📚 Research Background

This project applies established techniques from NLP and misinformation detection research:

1. **TF-IDF with Sublinear Scaling** — Dampens the impact of very frequent terms with `1 + log(tf)`, improving feature discrimination across large corpora.
2. **Ensemble Learning** — Voting classifiers reduce variance and increase robustness compared to any single model, a well-known technique in statistical learning.
3. **Probability Calibration** — Sigmoid (Platt scaling) calibration produces probabilities that reflect true class frequencies, critical for trustworthy confidence scores.
4. **Stylometric / Linguistic Analysis** — Fake news tends to exhibit specific surface-level cues: excessive capitalization, exclamation marks, sensational vocabulary, and clickbait sentence structures.
5. **Explainable AI (XAI)** — Feature-importance explanations let users evaluate predictions rather than blindly trusting a black box, aligning with responsible AI principles.
6. **Source-Attribution Debiasing** — Removing Reuters/AP bylines from training data prevents the model from learning shortcut features unrelated to content veracity.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright © 2026 Ghulam Mustafa

---

## 🙏 Acknowledgments

- **[Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)** by Clément Bisaillon — training data
- **[NLTK](https://www.nltk.org/)** — natural language processing toolkit
- **[scikit-learn](https://scikit-learn.org/)** — machine learning library
- **[Flask](https://flask.palletsprojects.com/)** — lightweight web framework
- Research community studying computational approaches to misinformation detection
