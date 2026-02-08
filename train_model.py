"""
Fake News Detection Model Training Script
Uses TF-IDF vectorization and SGD Classifier for best results
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            max_df=0.7,
            min_df=2,
            ngram_range=(1, 2)
        )
        # Using SGDClassifier with hinge loss (equivalent to linear SVM)
        self.model = SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=0.0001,
            max_iter=50,
            random_state=42
        )
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, texts, labels):
        """Train the fake news detection model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize text
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train model
        print("Training the model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        
        # Get decision function for confidence
        try:
            decision = self.model.decision_function(text_tfidf)[0]
            # Convert to confidence score (0-100%)
            confidence = min(100, max(0, 50 + abs(decision) * 10))
        except:
            confidence = 75.0  # Default confidence
        
        return prediction, confidence
    
    def save_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Save the trained model and vectorizer"""
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, vectorizer_path='vectorizer.pkl', model_path='model.pkl'):
        """Load a pre-trained model and vectorizer"""
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        print("Model and vectorizer loaded successfully!")


def create_sample_data():
    """Create sample training data for demonstration"""
    fake_news = [
        "Breaking: Aliens landed in New York City yesterday and nobody noticed!",
        "Scientists discover that drinking coffee can make you live forever",
        "Government confirms time travel is real and available for purchase",
        "New study shows that eating pizza prevents all diseases",
        "Local man discovers cure for aging using only household items",
        "Breaking news: Internet to be shut down permanently next week",
        "Shocking discovery: Moon is actually made of cheese confirms NASA",
        "Miracle cure discovered: Water can now cure all cancers",
        "Government hiding truth about flying cars already in use",
        "Scientists prove earth is actually flat using new technology",
        "Breaking: All birds are government drones used for surveillance",
        "New law requires everyone to change their name next month",
        "Study shows that looking at screens prevents blindness",
        "Local woman earns millions working just 5 minutes per day",
        "Breaking: Chocolate now classified as a vegetable by FDA",
        "Miracle diet lets you eat unlimited junk food and lose weight",
        "Secret society controls all world governments from underground base",
        "Drinking bleach cures all known diseases says anonymous doctor",
        "Scientists confirm: Santa Claus is real and lives in North Pole",
        "Breaking: Gravity no longer works in some parts of the world",
        "Government admits reptilians control the banking system",
        "New study: Sleeping with phone under pillow gives superpowers",
        "Breaking: Time traveler from 2050 warns of upcoming disaster",
        "Scientists create machine that turns water into gold instantly",
        "Government bans smiling in public places starting next month",
        "Miracle supplement allows humans to breathe underwater",
        "Breaking: Dinosaurs still alive and living in secret location",
        "New law requires all pets to attend mandatory school",
        "Scientists discover portal to alternate dimension in basement",
        "Breaking: Moon to be replaced with giant LED screen",
        "Study shows that vaccines contain microchips for mind control",
        "Government confirms existence of bigfoot and yeti",
        "New technology allows people to read minds using phone",
        "Breaking: Atlantis discovered and residents are using advanced tech",
        "Scientists prove that humans can photosynthesize like plants",
    ]
    
    real_news = [
        "Stock market shows mixed results in today's trading session",
        "Local community comes together to support new library opening",
        "Weather forecast predicts rain for the weekend ahead",
        "City council approves new budget for infrastructure improvements",
        "Research team publishes findings on climate change impacts",
        "New study suggests balanced diet and exercise improve health",
        "Technology company announces quarterly earnings report",
        "Local school district implements new educational program",
        "Scientists continue research on renewable energy sources",
        "Government officials discuss healthcare policy reforms",
        "University researchers make progress in medical research",
        "Local business expands operations creating new jobs",
        "Transportation department announces road maintenance schedule",
        "Community center offers new programs for senior citizens",
        "Environmental group works on conservation project",
        "Mayor announces plans for new public transportation system",
        "Research shows importance of early childhood education",
        "Local hospital receives funding for equipment upgrades",
        "City planning commission reviews zoning regulations",
        "Police department launches community outreach program",
        "School board approves new curriculum standards",
        "Local farmers market celebrates tenth anniversary",
        "County implements recycling program for residents",
        "University offers new scholarships for students",
        "City approves funding for park renovations",
        "Local organization provides meals for families in need",
        "Research indicates benefits of regular physical activity",
        "Community volunteers participate in neighborhood cleanup",
        "Library expands digital resources for patrons",
        "City council considers new affordable housing initiatives",
        "Local businesses partner with schools for mentorship program",
        "Health department offers free vaccination clinics",
        "University researchers study effects of air quality",
        "City invests in pedestrian safety improvements",
        "Local arts center announces upcoming exhibition schedule",
    ]
    
    # Create dataset
    texts = fake_news + real_news
    labels = [0] * len(fake_news) + [1] * len(real_news)  # 0=Fake, 1=Real
    
    return texts, labels


if __name__ == "__main__":
    print("=" * 60)
    print("Fake News Detection Model Training")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample training data...")
    texts, labels = create_sample_data()
    print(f"Dataset created with {len(texts)} samples")
    print(f"- Fake news samples: {labels.count(0)}")
    print(f"- Real news samples: {labels.count(1)}")
    
    # Initialize and train model
    detector = FakeNewsDetector()
    accuracy = detector.train(texts, labels)
    
    # Save the model
    detector.save_model()
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Testing Predictions:")
    print("=" * 60)
    
    test_texts = [
        "Scientists make breakthrough in cancer research at local university",
        "Aliens control the government and hide among us confirmed",
        "Local community celebrates new park opening with festival"
    ]
    
    for text in test_texts:
        prediction, confidence = detector.predict(text)
        label = "REAL" if prediction == 1 else "FAKE"
        print(f"\nText: {text}")
        print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
