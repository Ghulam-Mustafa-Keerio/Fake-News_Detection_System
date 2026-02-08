"""
Flask Web Application for Fake News Detection
"""

from flask import Flask, render_template, request, jsonify
import os
import joblib
from train_model import FakeNewsDetector, REAL_LABEL

app = Flask(__name__)

# Initialize detector
detector = FakeNewsDetector()

# Load pre-trained model if available
try:
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        detector.load_model()
        print("Pre-trained model loaded successfully!")
    else:
        print("No pre-trained model found. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading model: {e}")


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the news is fake or real"""
    try:
        # Get text from request
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Please provide some text to analyze'
            }), 400
        
        # Make prediction
        prediction, confidence = detector.predict(text)
        label = "REAL" if prediction == REAL_LABEL else "FAKE"
        
        # Prepare response
        result = {
            'prediction': label,
            'confidence': round(confidence, 2),
            'text': text[:200] + '...' if len(text) > 200 else text
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl')
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Fake News Detection Web Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Visit http://localhost:5000 to use the application")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
