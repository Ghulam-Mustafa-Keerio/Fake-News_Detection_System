"""
Tkinter GUI for Fake News Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
from train_model import AdvancedFakeNewsDetector as FakeNewsDetector, REAL_LABEL


class FakeNewsDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection System")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Initialize detector
        self.detector = FakeNewsDetector()
        self.model_loaded = False
        
        # Load model
        self.load_model()
        
        # Setup GUI
        self.setup_ui()
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
                self.detector.load_model()
                self.model_loaded = True
                print("Model loaded successfully!")
            else:
                messagebox.showwarning(
                    "Model Not Found",
                    "Pre-trained model not found. Please run train_model.py first.\n\n"
                    "The application will still work but needs training."
                )
                self.model_loaded = False
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
            self.model_loaded = False
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#2c3e50', pady=15)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="🔍 Fake News Detection System",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Using Advanced NLP Techniques",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack()
        
        # Status indicator
        status_text = "✓ Model Loaded" if self.model_loaded else "⚠ Model Not Loaded"
        status_color = '#27ae60' if self.model_loaded else '#e74c3c'
        
        status_label = tk.Label(
            title_frame,
            text=status_text,
            font=('Arial', 10),
            bg='#2c3e50',
            fg=status_color
        )
        status_label.pack(pady=5)
        
        # Main content frame
        content_frame = tk.Frame(self.root, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input section
        input_label = tk.Label(
            content_frame,
            text="Enter news text to analyze:",
            font=('Arial', 12, 'bold')
        )
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            content_frame,
            height=10,
            font=('Arial', 11),
            wrap=tk.WORD,
            borderwidth=2,
            relief=tk.GROOVE
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Sample text button
        sample_frame = tk.Frame(content_frame)
        sample_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            sample_frame,
            text="Load Sample Fake News",
            command=self.load_fake_sample
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            sample_frame,
            text="Load Sample Real News",
            command=self.load_real_sample
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            sample_frame,
            text="Clear",
            command=self.clear_text
        ).pack(side=tk.LEFT)
        
        # Analyze button
        self.analyze_button = tk.Button(
            content_frame,
            text="🔍 Analyze News",
            font=('Arial', 14, 'bold'),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor='hand2',
            pady=10,
            command=self.analyze_news
        )
        self.analyze_button.pack(fill=tk.X, pady=(0, 15))
        
        # Result frame
        result_frame = tk.LabelFrame(
            content_frame,
            text="Analysis Result",
            font=('Arial', 12, 'bold'),
            padx=15,
            pady=15
        )
        result_frame.pack(fill=tk.X)
        
        self.result_label = tk.Label(
            result_frame,
            text="No analysis yet. Enter text and click 'Analyze News'.",
            font=('Arial', 11),
            wraplength=700,
            justify=tk.LEFT
        )
        self.result_label.pack()
        
        # Footer
        footer_frame = tk.Frame(self.root, bg='#ecf0f1', pady=10)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_label = tk.Label(
            footer_frame,
            text="Powered by Machine Learning & NLP",
            font=('Arial', 9),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        footer_label.pack()
    
    def load_fake_sample(self):
        """Load a sample fake news text"""
        sample = ("Breaking News: Scientists have discovered that aliens are living "
                 "among us and controlling the world government. Anonymous sources "
                 "confirm that the moon landing was faked and the earth is actually "
                 "flat. This shocking revelation will change everything we know!")
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample)
    
    def load_real_sample(self):
        """Load a sample real news text"""
        sample = ("Local community celebrates the opening of a new public library. "
                 "The facility, which cost $2 million to build, will serve residents "
                 "with access to books, computers, and educational programs. "
                 "City officials and community members attended the ribbon-cutting "
                 "ceremony yesterday afternoon.")
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(1.0, sample)
    
    def clear_text(self):
        """Clear the text input"""
        self.text_input.delete(1.0, tk.END)
        self.result_label.config(
            text="No analysis yet. Enter text and click 'Analyze News'.",
            fg='black'
        )
    
    def analyze_news(self):
        """Analyze the input text for fake news"""
        if not self.model_loaded:
            messagebox.showerror(
                "Error",
                "Model not loaded. Please run train_model.py first."
            )
            return
        
        # Get input text
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        try:
            # Make prediction
            prediction, confidence = self.detector.predict(text)
            label = "REAL" if prediction == REAL_LABEL else "FAKE"
            
            # Update result
            if label == "FAKE":
                result_text = f"⚠️ FAKE NEWS DETECTED!\n\nConfidence: {confidence:.2f}%"
                color = '#e74c3c'
            else:
                result_text = f"✓ REAL NEWS\n\nConfidence: {confidence:.2f}%"
                color = '#27ae60'
            
            self.result_label.config(
                text=result_text,
                fg=color,
                font=('Arial', 14, 'bold')
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = FakeNewsDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
