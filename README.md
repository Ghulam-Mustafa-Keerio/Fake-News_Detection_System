# 🔍 Fake News Detection System

An AI-powered machine learning system for detecting fake news using Natural Language Processing (NLP) and Explainable AI. Built with Python, Flask, and scikit-learn.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Ensemble ML Model** | Combines multiple classifiers for robust predictions |
| **Explainable AI** | Shows *why* the model made its prediction |
| **Linguistic Analysis** | Detects sensational language, clickbait, and credibility indicators |
| **Risk Detection** | Identifies red flags common in fake news |
| **Web Interface** | Modern, responsive Flask-based application |
| **Desktop GUI** | Tkinter-based standalone application |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Web Application
```bash
python app.py
```

### 4. Open in Browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
Fake-News_Detection_System/
│
├── train_model.py      # Model training with NLP preprocessing
├── app.py              # Flask web application
├── gui.py              # Desktop GUI application
├── dataset_utils.py    # Dataset loading utilities
│
├── model.json          # Trained ML model (SECURE JSON format)
├── vectorizer.json     # TF-IDF vectorizer (SECURE JSON format)
│
├── templates/
│   ├── index.html      # Main web interface
│   └── about.html      # About page
│
└── requirements.txt    # Python dependencies
```

---

## 🔒 Security Features

**This project uses secure serialization (NO PICKLE FILES):**

| Risk | Our Solution |
|------|-------------|
| **Pickle Code Injection** | ❌ Eliminated - Using JSON/NumPy arrays only |
| **Data Tampering** | ✅ HMAC-SHA256 signature verification |
| **Model Integrity** | ✅ All files are human-readable JSON |

### Why Not Pickle?
Pickle files can execute arbitrary Python code when loaded, creating a serious security vulnerability:
```python
# DANGER: Pickle can run malicious code
import pickle
pickle.loads(malicious_data)  # Executes attacker's code!
```

### Our Secure Approach:
```python
# SAFE: JSON cannot execute code
import json
model_data = json.load(open('model.json'))  # Just data, no code execution
```

---

## 🧠 How It Works

### 1. Text Preprocessing
```
Raw Text → Lowercase → Remove URLs → Remove Special Chars → Lemmatization → Clean Text
```

### 2. Feature Extraction (TF-IDF)
- **10,000** vocabulary features
- **N-grams**: Unigrams, Bigrams, Trigrams
- **Sublinear TF scaling** for better term weighting

### 3. Ensemble Classification
```
Text → SGD Classifier    ──┐
Text → Logistic Regression─┼──→ Soft Voting → Final Prediction
Text → Random Forest     ──┘
```

### 4. Explainability Analysis
- **Top Features**: Which words influenced the prediction
- **Risk Factors**: Sensational language, clickbait patterns, ALL CAPS
- **Credibility Indicators**: Quotes, professional tone, content length

---

## 📊 Model Architecture

| Component | Configuration |
|-----------|---------------|
| **Vectorizer** | TF-IDF (10,000 features, n-gram 1-3) |
| **Classifier 1** | SGD with Modified Huber Loss |
| **Classifier 2** | Logistic Regression |
| **Classifier 3** | Random Forest (200 trees) |
| **Voting** | Soft voting ensemble |
| **Calibration** | Sigmoid probability calibration |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/predict` | POST | Analyze text and get prediction |
| `/analyze` | POST | Linguistic analysis only |
| `/health` | GET | Check model status |

### Example API Call
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article here"}'
```

---

## 📈 Sample Results

**Fake News Detection:**
```
Text: "BREAKING!!! Scientists CONFIRM aliens control the government!!!"
Prediction: FAKE (94% confidence)
Risk Factors:
  ⚠️ Clickbait patterns detected
  ⚠️ Excessive exclamation marks
  ⚠️ Multiple ALL CAPS words
```

**Real News Detection:**
```
Text: "The Federal Reserve announced it will maintain current interest rates."
Prediction: REAL (85% confidence)
Credibility Indicators:
  ✓ Neutral language tone
  ✓ Professional punctuation
  ✓ Appropriate sentence complexity
```

---

## 🔧 Technologies Used

- **Python 3.8+**
- **Flask** - Web framework
- **scikit-learn** - Machine learning
- **NLTK** - Natural language processing
- **TF-IDF** - Feature extraction
- **Ensemble Learning** - Model combination

---

## 📚 Research Background

This project implements techniques from NLP research:

1. **TF-IDF with Sublinear Scaling** - Better term frequency handling
2. **Ensemble Learning** - Combining classifiers for robustness
3. **Probability Calibration** - Reliable confidence scores
4. **Linguistic Feature Analysis** - Stylometric fake news detection
5. **Explainable AI** - Feature importance visualization

---

## 👨‍💻 Usage Examples

### Python API
```python
from train_model import AdvancedFakeNewsDetector

# Load model
detector = AdvancedFakeNewsDetector()
detector.load_model()

# Make prediction
result = detector.predict("Your news text here", explain=True)
print(f"Prediction: {result.label}")
print(f"Confidence: {result.confidence}%")
print(f"Risk Factors: {result.risk_factors}")
```

### Command Line
```bash
# Train model
python train_model.py

# Start web server
python app.py

# Start desktop GUI
python gui.py
```

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- NLTK for natural language processing tools
- scikit-learn for machine learning algorithms
- Flask for the web framework
- Research papers on fake news detection and explainable AI
