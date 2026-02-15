"""
Dataset Utilities for Fake News Detection
Provides functions to download and prepare real datasets for training

Supported Datasets:
1. Kaggle Fake and Real News Dataset (~45k articles)
2. LIAR Dataset (politifact.com fact-checks)
3. Custom CSV datasets
"""

import os
import pandas as pd
import urllib.request
import zipfile
from typing import Tuple, List, Optional


def download_sample_dataset() -> Tuple[List[str], List[int]]:
    """
    Download a sample dataset for demonstration
    
    Returns:
        Tuple of (texts, labels) where labels are 0=Fake, 1=Real
    """
    print("Creating extended sample dataset...")
    
    fake_news = [
        # Conspiracy theories - 30 examples
        "Breaking: Aliens landed in New York City yesterday and nobody noticed!",
        "Scientists discover that drinking coffee can make you live forever",
        "Government confirms time travel is real and available for purchase",
        "New study shows that eating pizza prevents all diseases",
        "Local man discovers cure for aging using only household items",
        "Breaking news: Internet to be shut down permanently next week",
        "Shocking discovery: Moon is actually made of cheese confirms NASA",
        "Miracle cure discovered: Water can now cure all cancers",
        "Scientists prove earth is actually flat using new technology",
        "Breaking: All birds are government drones used for surveillance",
        "Study shows that looking at screens prevents blindness",
        "Breaking: Chocolate now classified as a vegetable by FDA",
        "Miracle diet lets you eat unlimited junk food and lose weight",
        "Secret society controls all world governments from underground base",
        "Scientists confirm: Santa Claus is real and lives in North Pole",
        "Breaking: Gravity no longer works in some parts of the world",
        "Government admits reptilians control the banking system",
        "Scientists create machine that turns water into gold instantly",
        "Government bans smiling in public places starting next month",
        "Miracle supplement allows humans to breathe underwater",
        "Breaking: Dinosaurs still alive and living in secret location",
        "Scientists discover portal to alternate dimension in basement",
        "Breaking: Moon to be replaced with giant LED screen",
        "Study shows that vaccines contain microchips for mind control",
        "Government confirms existence of bigfoot and yeti",
        "Breaking: Atlantis discovered with advanced technology",
        "Scientists prove that humans can photosynthesize like plants",
        "New law makes it illegal to think negative thoughts",
        "Breaking: World leaders are actually robots in disguise",
        "Scientists discover that the sun is actually cold inside",
        
        # Clickbait/Sensational - 20 examples
        "YOU WON'T BELIEVE what this celebrity did yesterday!!!",
        "SHOCKING: What happens next will blow your mind completely",
        "Doctors HATE this one weird trick to lose 50 pounds overnight",
        "BREAKING: This common household item is secretly killing you",
        "EXPOSED: The truth they don't want you to know about water",
        "URGENT: Share this before it gets deleted by the government",
        "MIRACLE: Man grows 10 inches taller using this simple method",
        "SECRET: Banks don't want you to know this money trick",
        "ALERT: Your smartphone is spying on you right now confirmed",
        "SCANDAL: Famous politician caught shapeshifting on camera",
        "This weird trick will make you a millionaire in just 3 days!!!",
        "What this mother found in her child's lunchbox will SHOCK you",
        "Scientists are BAFFLED by this discovery that changes everything",
        "The government doesn't want you to see this video about aliens",
        "This one food that doctors say you should NEVER eat again",
        "How this man made $50,000 in one week from home REVEALED",
        "The SHOCKING truth about what's really in your tap water",
        "What they found in this celebrity's house will DISTURB you",
        "This ancient secret will transform your life overnight GUARANTEED",
        "Why you should STOP doing this immediately according to experts",
        
        # Health misinformation - 15 examples
        "New research proves vaccines cause autism in all children",
        "5G towers confirmed to spread coronavirus by government leak",
        "Eating dirt daily cures depression according to new study",
        "Scientists discover that sunlight causes cancer instantly",
        "New pill allows you to never sleep again without side effects",
        "Drinking your own urine cures all diseases says doctor",
        "Essential oils proven to cure cancer in clinical trials",
        "Hospitals hiding the fact that sugar cures diabetes",
        "Meditation can regrow lost limbs according to new research",
        "Coffee proven to cause heart attacks in everyone who drinks it",
        "Vegans live 100 years longer than meat eaters study shows",
        "Chemotherapy is a poison that kills more people than cancer",
        "Natural remedies Big Pharma doesn't want you to know about",
        "This miracle berry cures blindness in just 24 hours",
        "Wearing masks causes oxygen deprivation and brain damage",
    ]
    
    real_news = [
        # General/Political news - 25 examples
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
        
        # Technology news - 15 examples
        "Tech company releases software update fixing security issues",
        "University researchers develop new algorithm for data processing",
        "Conference brings together experts to discuss AI ethics guidelines",
        "Startup receives funding to develop sustainable technology solutions",
        "Report shows increase in remote work adoption across industries",
        "New cybersecurity guidelines released for businesses",
        "Scientists test quantum computing applications for research",
        "Industry report analyzes trends in renewable energy technology",
        "Government proposes new regulations for data privacy protection",
        "Tech conference focuses on accessibility in software design",
        "Researchers publish findings on machine learning applications",
        "Cloud computing adoption increases among small businesses",
        "University launches program to train workers in digital skills",
        "Experts discuss implications of automation in manufacturing",
        "New standards proposed for electric vehicle infrastructure",
        
        # Science news - 15 examples
        "Researchers publish peer-reviewed study on sleep patterns",
        "Scientists observe new species in deep sea expedition",
        "University team develops more efficient solar panel design",
        "Study examines relationship between exercise and mental health",
        "International collaboration advances understanding of genetics",
        "Astronomers discover new exoplanets using telescope data",
        "Climate scientists present findings at international conference",
        "Research team studies effects of pollution on marine ecosystems",
        "New archaeological discoveries shed light on ancient civilizations",
        "Scientists develop new method for testing water quality",
        "Study analyzes long-term effects of urban development",
        "Researchers investigate potential treatments for rare diseases",
        "Field study documents migration patterns of endangered species",
        "University publishes research on sustainable agriculture practices",
        "Geologists study seismic activity to improve earthquake prediction",
        
        # Business/Economy news - 10 examples
        "Company announces expansion plans for next fiscal year",
        "Industry report shows steady growth in manufacturing sector",
        "Small businesses adapt to changing consumer preferences",
        "Trade agreement negotiations continue between nations",
        "Analysts predict moderate economic growth for upcoming quarter",
        "Federal Reserve announces decision on interest rates",
        "Employment statistics show stable job market conditions",
        "Consumer spending increases during holiday shopping season",
        "Central bank releases annual economic outlook report",
        "International trade volume remains consistent with projections",
    ]
    
    texts = fake_news + real_news
    labels = [0] * len(fake_news) + [1] * len(real_news)
    
    print(f"Dataset created: {len(texts)} samples ({len(fake_news)} fake, {len(real_news)} real)")
    
    return texts, labels


def load_kaggle_dataset(path: str = "data/fake_and_real_news/") -> Tuple[List[str], List[int]]:
    """
    Load the Kaggle Fake and Real News Dataset
    
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    
    Args:
        path: Directory containing Fake.csv and True.csv
        
    Returns:
        Tuple of (texts, labels)
    """
    fake_path = os.path.join(path, "Fake.csv")
    true_path = os.path.join(path, "True.csv")

    # If files are not found in the provided path, try common alternatives
    if not (os.path.exists(fake_path) and os.path.exists(true_path)):
        alt_fake = os.path.join(os.getcwd(), "Fake.csv")
        alt_true = os.path.join(os.getcwd(), "True.csv")
        if os.path.exists(alt_fake) and os.path.exists(alt_true):
            fake_path, true_path = alt_fake, alt_true
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}.\n"
                "Please download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
                "And extract Fake.csv and True.csv to the specified directory (or run the provided download script)."
            )
    
    print("Loading Kaggle Fake and Real News Dataset...")
    
    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Strip Reuters/AP source-attribution patterns from True.csv texts
    # (e.g. "WASHINGTON (Reuters) - ...", "(Reuters) - ...", city names with
    # commas/abbreviations, correction notes, etc.) so the model learns
    # content features, not source-line shortcuts.
    true_df['text'] = true_df['text'].str.replace(
        r'^.*?\(Reuters\)\s*-\s*', '', regex=True
    )
    true_df['text'] = true_df['text'].str.replace(
        r'^.*?\(AP\)\s*-\s*', '', regex=True
    )
    # Strip correction/editor notes at start
    true_df['text'] = true_df['text'].str.replace(
        r'^\s*\(In .+? story,.*?\)\s*', '', regex=True
    )
    # Strip "The following statements were posted to the verified Twitter accounts of..."
    true_df['text'] = true_df['text'].str.replace(
        r'^The following statements were posted to the verified Twitter accounts? of U\.S\. President Donald Trump.*?$',
        '', regex=True, flags=0
    )
    
    # Combine title and text
    fake_df['content'] = fake_df['title'] + " " + fake_df['text']
    true_df['content'] = true_df['title'] + " " + true_df['text']
    
    # Create labels
    fake_df['label'] = 0
    true_df['label'] = 1
    
    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    texts = df['content'].tolist()
    labels = df['label'].tolist()
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"  Fake news: {labels.count(0)}")
    print(f"  Real news: {labels.count(1)}")
    
    return texts, labels


def load_csv_dataset(
    path: str, 
    text_column: str = "text",
    label_column: str = "label",
    fake_label: any = 0,
    real_label: any = 1
) -> Tuple[List[str], List[int]]:
    """
    Load a custom CSV dataset
    
    Args:
        path: Path to CSV file
        text_column: Name of the text/content column
        label_column: Name of the label column
        fake_label: Value representing fake news
        real_label: Value representing real news
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading dataset from {path}...")
    
    df = pd.read_csv(path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")
    
    # Convert labels to 0/1
    df['_label'] = df[label_column].apply(
        lambda x: 0 if x == fake_label else (1 if x == real_label else -1)
    )
    
    # Filter out invalid labels
    df = df[df['_label'] >= 0]
    
    texts = df[text_column].astype(str).tolist()
    labels = df['_label'].tolist()
    
    print(f"Dataset loaded: {len(texts)} samples")
    
    return texts, labels


def prepare_dataset_stats(texts: List[str], labels: List[int]) -> dict:
    """
    Compute statistics about a dataset
    """
    import numpy as np
    
    fake_texts = [t for t, l in zip(texts, labels) if l == 0]
    real_texts = [t for t, l in zip(texts, labels) if l == 1]
    
    def text_stats(text_list):
        if not text_list:
            return {}
        lengths = [len(t.split()) for t in text_list]
        return {
            'count': len(text_list),
            'avg_words': np.mean(lengths),
            'min_words': min(lengths),
            'max_words': max(lengths),
            'std_words': np.std(lengths)
        }
    
    return {
        'total_samples': len(texts),
        'fake_news': text_stats(fake_texts),
        'real_news': text_stats(real_texts),
        'balance_ratio': len(fake_texts) / max(len(real_texts), 1)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Utilities for Fake News Detection")
    print("=" * 60)
    
    # Demo with sample dataset
    texts, labels = download_sample_dataset()
    stats = prepare_dataset_stats(texts, labels)
    
    print("\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Fake news: {stats['fake_news']['count']}")
    print(f"  Real news: {stats['real_news']['count']}")
    print(f"  Balance ratio: {stats['balance_ratio']:.2f}")
    print(f"  Avg words (fake): {stats['fake_news']['avg_words']:.1f}")
    print(f"  Avg words (real): {stats['real_news']['avg_words']:.1f}")
    
    print("\n" + "=" * 60)
    print("To use a real dataset for better accuracy:")
    print("=" * 60)
    print("""
1. Kaggle Fake and Real News Dataset (Recommended):
   - Download: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
   - Extract to: data/fake_and_real_news/
   - Use: texts, labels = load_kaggle_dataset("data/fake_and_real_news/")

2. Custom CSV Dataset:
   - Use: texts, labels = load_csv_dataset("your_data.csv", 
                                           text_column="content",
                                           label_column="label",
                                           fake_label="fake",
                                           real_label="real")

3. Then train:
   from train_model_advanced import AdvancedFakeNewsDetector, ModelType
   detector = AdvancedFakeNewsDetector(ModelType.TFIDF_ENSEMBLE)
   detector.train(texts, labels)
   detector.save_model()
""")
