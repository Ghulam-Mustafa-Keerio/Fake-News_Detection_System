"""
Advanced Flask Web Application for Fake News Detection
Features:
- Explainable AI predictions
- Multiple model support (Ensemble, SGD, Logistic, Transformer)
- Confidence calibration
- Linguistic analysis
- Risk factor detection
- API endpoints for integration
"""

from flask import Flask, render_template, request, jsonify, abort
import os
import sys

# Import the fake news detection model
from train_model import (
    AdvancedFakeNewsDetector, 
    ModelType, 
    PredictionResult,
    REAL_LABEL, 
    FAKE_LABEL
)
ADVANCED_MODEL_AVAILABLE = True

# Try to import transformer model
try:
    from train_model_transformer import TransformerFakeNewsDetector, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

app = Flask(__name__)

# Initialize detector
detector = None
model_info = {
    'type': 'unknown',
    'loaded': False,
    'accuracy': None
}


def initialize_model():
    """Initialize the best available model"""
    global detector, model_info
    
    # Priority 1: Try advanced model (SECURE JSON format)
    if ADVANCED_MODEL_AVAILABLE:
        advanced_model_exists = (
            os.path.exists('model.json') and 
            os.path.exists('vectorizer.json')
        )
        
        if advanced_model_exists:
            try:
                detector = AdvancedFakeNewsDetector()
                detector.load_model()
                model_info = {
                    'type': 'advanced_ensemble',
                    'loaded': True,
                    'accuracy': detector.training_metrics.get('accuracy'),
                    'features': ['explainability', 'calibrated_confidence', 'linguistic_analysis'],
                    'security': 'json_only_no_pickle'
                }
                print("[SECURE] Advanced ensemble model loaded successfully!")
                return
            except Exception as e:
                print(f"Error loading advanced model: {e}")
    
    # Priority 2: Try basic model
    basic_model_exists = (
        os.path.exists('model.json') and 
        os.path.exists('vectorizer.json')
    )
    
    if basic_model_exists:
        try:
            if ADVANCED_MODEL_AVAILABLE:
                # Use advanced detector with basic model files
                detector = AdvancedFakeNewsDetector()
                detector.load_model()
            else:
                from train_model import FakeNewsDetector
                detector = FakeNewsDetector()
                detector.load_model()
            
            model_info = {
                'type': 'basic',
                'loaded': True,
                'accuracy': None,
                'features': ['basic_prediction']
            }
            print("Basic model loaded successfully!")
            return
        except Exception as e:
            print(f"Error loading basic model: {e}")
    
    print("⚠️  No pre-trained model found!")
    print("Please run one of the following:")
    print("  python train_model.py           (basic model)")
    print("  python train_model_advanced.py  (advanced model with explainability)")
    model_info['loaded'] = False


# Initialize on startup
initialize_model()


@app.route('/')
def home():
    """Render the home page"""
    # Use advanced template if it exists
    if os.path.exists('templates/index_advanced.html'):
        return render_template('index_advanced.html')
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the news is fake or real with explainability"""
    global detector, model_info
    
    if not model_info.get('loaded', False) or detector is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py or train_model_advanced.py first.'
        }), 503
    
    try:
        # Get text from request
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
            include_explanation = data.get('explain', True)
        else:
            text = request.form.get('text', '')
            include_explanation = request.form.get('explain', 'true').lower() == 'true'
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Please provide some text to analyze'
            }), 400
        
        # Make prediction with explainability if available
        if ADVANCED_MODEL_AVAILABLE and hasattr(detector, 'predict') and include_explanation:
            try:
                result = detector.predict(text, explain=True)
                
                if isinstance(result, PredictionResult):
                    response = {
                        'prediction': result.label,
                        'confidence': round(result.confidence, 1),
                        'probabilities': {
                            'fake': round(result.probabilities['Fake'], 1),
                            'real': round(result.probabilities['Real'], 1)
                        },
                        'text': text[:200] + '...' if len(text) > 200 else text,
                        'explanation': {
                            'top_features': [
                                {'word': f, 'weight': round(w, 4)} 
                                for f, w in result.top_features[:7]
                            ],
                            'linguistic_analysis': {
                                'word_count': result.linguistic_analysis.get('word_count', 0),
                                'sentence_count': result.linguistic_analysis.get('sentence_count', 0),
                                'sensational_words': result.linguistic_analysis.get('sensational_word_count', 0),
                                'exclamation_marks': result.linguistic_analysis.get('exclamation_count', 0),
                                'caps_ratio': round(result.linguistic_analysis.get('caps_ratio', 0) * 100, 1)
                            },
                            'risk_factors': result.risk_factors,
                            'credibility_indicators': result.credibility_indicators
                        },
                        'model_type': model_info.get('type', 'unknown')
                    }
                    return jsonify(response)
                else:
                    # Tuple result (prediction, confidence)
                    prediction, confidence = result
                    label = "REAL" if prediction == REAL_LABEL else "FAKE"
            except Exception as e:
                # Fallback to simple prediction
                prediction, confidence = detector.predict(text)
                label = "REAL" if prediction == REAL_LABEL else "FAKE"
        else:
            # Basic prediction
            prediction, confidence = detector.predict(text)
            label = "REAL" if prediction == REAL_LABEL else "FAKE"
        
        # Basic response
        result = {
            'prediction': label,
            'confidence': round(confidence, 1) if isinstance(confidence, float) else confidence,
            'text': text[:200] + '...' if len(text) > 200 else text,
            'model_type': model_info.get('type', 'unknown')
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple texts"""
    global detector, model_info
    
    if not model_info.get('loaded', False) or detector is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({'error': 'Please provide a list of texts'}), 400
        
        if len(texts) > 100:
            return jsonify({'error': 'Maximum 100 texts per batch'}), 400
        
        results = []
        for text in texts:
            try:
                prediction, confidence = detector.predict(text) if not ADVANCED_MODEL_AVAILABLE else detector.predict(text, explain=False)
                label = "REAL" if prediction == REAL_LABEL else "FAKE"
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'prediction': label,
                    'confidence': round(confidence, 1)
                })
            except:
                results.append({
                    'text': text[:100],
                    'error': 'Prediction failed'
                })
        
        return jsonify({
            'results': results,
            'total': len(results)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Detailed linguistic analysis without prediction"""
    if not ADVANCED_MODEL_AVAILABLE:
        return jsonify({'error': 'Advanced model required for analysis'}), 501
    
    try:
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Please provide text'}), 400
        
        from train_model_advanced import AdvancedTextPreprocessor
        preprocessor = AdvancedTextPreprocessor()
        
        features = preprocessor.extract_linguistic_features(text)
        risks = preprocessor.identify_risk_factors(features)
        credibility = preprocessor.identify_credibility_indicators(features)
        
        return jsonify({
            'linguistic_features': features,
            'risk_factors': risks,
            'credibility_indicators': credibility
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/health')
def health():
    """Health check endpoint with model info"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_info.get('loaded', False),
        'model_type': model_info.get('type', 'none'),
        'features': model_info.get('features', []),
        'advanced_available': ADVANCED_MODEL_AVAILABLE,
        'transformers_available': TRANSFORMERS_AVAILABLE
    })


@app.route('/api/models')
def list_models():
    """List available models and their status"""
    models = []
    
    # Check basic model
    if os.path.exists('model.json'):
        models.append({
            'name': 'Basic TF-IDF + SGD',
            'file': 'model.json',
            'status': 'available',
            'security': 'json_only_no_pickle'
        })
    
    # Check advanced model
    if os.path.exists('model.json'):
        models.append({
            'name': 'Advanced Ensemble',
            'file': 'model.json',
            'status': 'available',
            'features': ['explainability', 'calibration', 'linguistic_analysis'],
            'security': 'json_only_no_pickle'
        })
    
    # Check transformer model
    if os.path.exists('transformer_model'):
        models.append({
            'name': 'Transformer (Fine-tuned)',
            'file': 'transformer_model/',
            'status': 'available',
            'features': ['state-of-the-art', 'contextual_embeddings']
        })
    
    return jsonify({
        'models': models,
        'current': model_info.get('type', 'none')
    })


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining (admin endpoint)"""
    # This would typically require authentication
    return jsonify({
        'message': 'Retraining endpoint - implement with proper authentication',
        'status': 'not_implemented'
    }), 501


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Endpoint not found'}), 404
    return render_template('index.html')


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return "An error occurred. Please try again.", 500


if __name__ == '__main__':
    print("=" * 60)
    print("  ADVANCED FAKE NEWS DETECTION WEB APPLICATION")
    print("=" * 60)
    
    if model_info.get('loaded'):
        print(f"\n✓ Model Type: {model_info.get('type')}")
        if model_info.get('features'):
            print(f"✓ Features: {', '.join(model_info.get('features', []))}")
    else:
        print("\n⚠️  No model loaded!")
        print("   Run 'python train_model_advanced.py' to train the advanced model")
        print("   Or run 'python train_model.py' for the basic model")
    
    print("\n" + "-" * 60)
    print("Starting Flask server...")
    print("Visit http://localhost:5000 to use the application")
    print("-" * 60)
    print("\nAPI Endpoints:")
    print("  POST /predict       - Single text prediction")
    print("  POST /predict/batch - Batch predictions (max 100)")
    print("  POST /analyze       - Linguistic analysis only")
    print("  GET  /health        - Health check & model info")
    print("  GET  /api/models    - List available models")
    print("-" * 60)
    
    # Only enable debug mode if explicitly requested
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
