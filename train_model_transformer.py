"""
Transformer-Based Fake News Detector
Uses pre-trained RoBERTa/BERT models for state-of-the-art performance

This module provides high-accuracy fake news detection using transformer models
from Hugging Face. It requires additional dependencies: transformers, torch

Installation:
    pip install transformers torch

Usage:
    from train_model_transformer import TransformerFakeNewsDetector
    detector = TransformerFakeNewsDetector()
    detector.load_pretrained()  # Load pre-trained model from HuggingFace
    result = detector.predict("Your news text here")
"""

import os
import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Check if transformers is available
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        pipeline,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from torch.utils.data import Dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class TransformerPrediction:
    """Structured prediction result from transformer model"""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    all_scores: List[Dict[str, any]]


class FakeNewsDataset:
    """Custom dataset for fine-tuning transformers"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


class TransformerFakeNewsDetector:
    """
    Transformer-based fake news detector using HuggingFace models
    
    Supports:
    - Loading pre-trained fake news detection models
    - Fine-tuning on custom data
    - Zero-shot classification
    """
    
    # Pre-trained models from HuggingFace for fake news detection
    PRETRAINED_MODELS = {
        'roberta-fake-news': 'hamzab/roberta-fake-news-classification',
        'bert-fake-news': 'Narrativa/bert-tiny-finetuned-fake-news-detection',
        'distilbert-base': 'distilbert-base-uncased',
        'roberta-base': 'roberta-base'
    }
    
    def __init__(self, model_name: str = 'roberta-fake-news'):
        """
        Initialize the transformer detector
        
        Args:
            model_name: One of the pre-trained models or a HuggingFace model ID
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for this module.\n"
                "Install with: pip install transformers torch\n"
                "Or use the ensemble model in train_model_advanced.py instead."
            )
        
        self.model_name = model_name
        self.model_id = self.PRETRAINED_MODELS.get(model_name, model_name)
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_loaded = False
        self.label_mapping = {'LABEL_0': 'Fake', 'LABEL_1': 'Real', 'Fake': 'Fake', 'Real': 'Real'}
        
    def load_pretrained(self, use_pipeline: bool = True) -> None:
        """
        Load a pre-trained model from HuggingFace
        
        Args:
            use_pipeline: If True, creates a pipeline for easy inference
        """
        print(f"Loading transformer model: {self.model_id}")
        print(f"Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            
            if use_pipeline:
                self.pipeline = pipeline(
                    'text-classification',
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == 'cuda' else -1,
                    top_k=None  # Return all scores
                )
            
            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}!")
            
            # Update label mapping based on model config
            if hasattr(self.model.config, 'id2label'):
                self.label_mapping.update(self.model.config.id2label)
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative model...")
            self._try_alternative_model()
    
    def _try_alternative_model(self):
        """Try loading an alternative model if primary fails"""
        alternative = 'distilbert-base-uncased'
        print(f"Attempting to load alternative: {alternative}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(alternative)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                alternative, 
                num_labels=2
            )
            self.model.to(self.device)
            self.is_loaded = True
            print("Alternative model loaded. Note: This model needs fine-tuning for best results.")
        except Exception as e:
            raise RuntimeError(f"Could not load any transformer model: {e}")
    
    def predict(self, text: str) -> TransformerPrediction:
        """
        Make a prediction on the input text
        
        Args:
            text: The news article text to classify
            
        Returns:
            TransformerPrediction with label, confidence, and probabilities
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        # Format text for the model (some models expect specific format)
        if 'hamzab' in self.model_id:
            # This model expects: <title> TITLE <content> CONTENT <end>
            # If no title provided, we'll treat the first sentence as title
            sentences = text.split('.')
            if len(sentences) > 1:
                title = sentences[0].strip()
                content = '. '.join(sentences[1:]).strip()
            else:
                title = text[:50]
                content = text
            formatted_text = f"<title> {title} <content> {content} <end>"
        else:
            formatted_text = text
        
        try:
            if self.pipeline:
                results = self.pipeline(formatted_text[:512])  # Truncate to max length
                
                # Handle different return formats
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]
                
                # Process results
                all_scores = []
                probabilities = {'Fake': 0.0, 'Real': 0.0}
                
                for item in results:
                    label = item['label']
                    score = item['score']
                    
                    # Map label to Fake/Real
                    mapped_label = self.label_mapping.get(label, label)
                    if 'fake' in mapped_label.lower() or label == 'LABEL_0':
                        probabilities['Fake'] = score * 100
                    else:
                        probabilities['Real'] = score * 100
                    
                    all_scores.append({
                        'label': mapped_label,
                        'score': score
                    })
                
                # Determine final label
                if probabilities['Fake'] > probabilities['Real']:
                    final_label = 'FAKE'
                    confidence = probabilities['Fake']
                else:
                    final_label = 'REAL'
                    confidence = probabilities['Real']
                
                return TransformerPrediction(
                    label=final_label,
                    confidence=confidence,
                    probabilities=probabilities,
                    all_scores=all_scores
                )
            
            else:
                # Direct model inference
                inputs = self.tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0]
                
                # Assuming label 0 = Fake, 1 = Real
                fake_prob = probs[0].item() * 100
                real_prob = probs[1].item() * 100
                
                if fake_prob > real_prob:
                    final_label = 'FAKE'
                    confidence = fake_prob
                else:
                    final_label = 'REAL'
                    confidence = real_prob
                
                return TransformerPrediction(
                    label=final_label,
                    confidence=confidence,
                    probabilities={'Fake': fake_prob, 'Real': real_prob},
                    all_scores=[
                        {'label': 'Fake', 'score': fake_prob / 100},
                        {'label': 'Real', 'score': real_prob / 100}
                    ]
                )
                
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, texts: List[str]) -> List[TransformerPrediction]:
        """
        Make predictions on multiple texts
        
        Args:
            texts: List of news article texts
            
        Returns:
            List of TransformerPrediction objects
        """
        return [self.predict(text) for text in texts]
    
    def fine_tune(
        self, 
        train_texts: List[str], 
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        output_dir: str = './transformer_model'
    ) -> Dict:
        """
        Fine-tune the model on custom data
        
        Args:
            train_texts: Training text samples
            train_labels: Training labels (0=Fake, 1=Real)
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_dir: Directory to save the fine-tuned model
            
        Returns:
            Training metrics
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_pretrained() first.")
        
        print(f"\nFine-tuning transformer model on {len(train_texts)} samples...")
        
        # Prepare datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer)
        
        val_dataset = None
        if val_texts and val_labels:
            val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\nFine-tuned model saved to {output_dir}")
        
        return {
            'training_loss': train_result.training_loss,
            'epochs': epochs,
            'samples': len(train_texts)
        }
    
    def save_model(self, path: str = './transformer_model'):
        """Save the current model to disk"""
        if not self.is_loaded:
            raise ValueError("No model loaded to save.")
        
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'model_id': self.model_id,
            'label_mapping': self.label_mapping
        }
        with open(os.path.join(path, 'detector_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str = './transformer_model'):
        """Load a saved model from disk"""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load config if available
        config_path = os.path.join(path, 'detector_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.label_mapping.update(config.get('label_mapping', {}))
        
        self.pipeline = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1,
            top_k=None
        )
        
        self.is_loaded = True
        print(f"Model loaded from {path}")


def demo():
    """Demo of transformer-based fake news detection"""
    print("=" * 70)
    print("   TRANSFORMER-BASED FAKE NEWS DETECTION DEMO")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  Transformers library not installed!")
        print("\nTo use transformer models, install the required packages:")
        print("    pip install transformers torch")
        print("\nAlternatively, use the ensemble model in train_model_advanced.py")
        return
    
    # Initialize with pre-trained RoBERTa model
    print("\nLoading pre-trained RoBERTa fake news classifier...")
    detector = TransformerFakeNewsDetector('roberta-fake-news')
    
    try:
        detector.load_pretrained()
    except Exception as e:
        print(f"\n⚠️  Could not load pre-trained model: {e}")
        print("This may require internet connection to download the model.")
        return
    
    # Test cases
    test_cases = [
        {
            "text": "Scientists at Stanford University have completed a comprehensive 10-year study examining the effects of regular physical exercise on cardiovascular health. The peer-reviewed study, published in the Journal of Medicine, analyzed data from 50,000 participants and found significant improvements in heart health among those who exercised regularly.",
            "expected": "REAL"
        },
        {
            "text": "BREAKING NEWS!!! You won't BELIEVE what scientists discovered this morning! Proof that ALIENS have been controlling world governments for DECADES! Multiple sources confirm this SHOCKING revelation. Share before this gets DELETED!!!",
            "expected": "FAKE"
        },
        {
            "text": "The Federal Reserve announced today that it will maintain current interest rates, citing stable inflation levels and moderate economic growth. The decision was widely anticipated by financial analysts and follows the central bank's cautious approach to monetary policy.",
            "expected": "REAL"
        }
    ]
    
    print("\n" + "-" * 70)
    print("TESTING TRANSFORMER MODEL")
    print("-" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"Text: {case['text'][:100]}...")
        print(f"Expected: {case['expected']}")
        
        result = detector.predict(case['text'])
        
        print(f"Prediction: {result.label}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"Probabilities: Fake={result.probabilities['Fake']:.1f}%, Real={result.probabilities['Real']:.1f}%")
        
        status = "✓ CORRECT" if result.label == case['expected'] else "✗ INCORRECT"
        print(f"Status: {status}")
    
    print("\n" + "=" * 70)
    print("   DEMO COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    demo()
