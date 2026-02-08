# 🔍 Fake News Detection System

An advanced machine learning application that uses Natural Language Processing (NLP) techniques to detect fake news articles. The system provides both a web interface and a desktop GUI for easy testing and deployment.

## ✨ Features

- **Advanced NLP Analysis**: Uses TF-IDF vectorization and Passive Aggressive Classifier
- **Web Application**: Flask-based web interface accessible from any device
- **Desktop GUI**: Tkinter-based graphical interface for standalone use
- **Real-time Predictions**: Instant analysis with confidence scores
- **Sample Data**: Pre-loaded examples for testing
- **Text Preprocessing**: Comprehensive cleaning including stopword removal and stemming

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Ghulam-Mustafa-Keerio/Fake-News_Detection_System.git
cd Fake-News_Detection_System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

This will create `model.pkl` and `vectorizer.pkl` files needed for predictions.

## 💻 Usage

### Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter news text and click "Analyze News" to get predictions.

### Desktop GUI (Tkinter)

Run the Tkinter interface:
```bash
python gui.py
```

The GUI provides:
- Text input area for news articles
- Sample data buttons for testing
- Real-time analysis with visual feedback
- Confidence scores for predictions

## 🧠 How It Works

### 1. Text Preprocessing
- Converts text to lowercase
- Removes URLs, mentions, and special characters
- Eliminates stopwords
- Applies stemming for normalization

### 2. Feature Extraction
- Uses TF-IDF (Term Frequency-Inverse Document Frequency)
- Captures word importance in context
- Generates numerical features from text

### 3. Classification
- Passive Aggressive Classifier for real-time learning
- Trained on labeled fake/real news dataset
- Provides binary classification with confidence scores

### 4. Model Performance
The model uses:
- **TF-IDF Vectorizer**: max_features=5000, ngram_range=(1,2)
- **Passive Aggressive Classifier**: max_iter=50
- **Train/Test Split**: 80/20 ratio

## 📁 Project Structure

```
Fake-News_Detection_System/
├── train_model.py          # Model training script
├── app.py                  # Flask web application
├── gui.py                  # Tkinter desktop GUI
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── templates/             # HTML templates
│   ├── index.html        # Main web interface
│   └── about.html        # About page
├── model.pkl             # Trained model (generated)
└── vectorizer.pkl        # TF-IDF vectorizer (generated)
```

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework for REST API
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing toolkit
- **Pandas & NumPy**: Data manipulation
- **Tkinter**: Desktop GUI framework
- **Joblib**: Model serialization

## 📊 Sample Usage

### Python API Example

```python
from train_model import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()
detector.load_model()

# Make prediction
text = "Your news article text here..."
prediction, confidence = detector.predict(text)

if prediction == 1:
    print(f"REAL NEWS (Confidence: {confidence:.2f}%)")
else:
    print(f"FAKE NEWS (Confidence: {confidence:.2f}%)")
```

## ⚠️ Important Notes

- This system is an assistive tool, not a definitive fact-checker
- Always verify important news from multiple reliable sources
- Model accuracy depends on training data quality and diversity
- The system works best with English language news articles

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with open-source machine learning libraries
- Inspired by the need to combat misinformation
- Thanks to the NLP and ML communities for their tools and resources

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating NLP and machine learning applications. For production use, consider using larger, more diverse training datasets and additional validation mechanisms.
