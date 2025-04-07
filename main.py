import os
import json
import pandas as pd
import spacy
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Load models
nlp_spacy = spacy.load("en_core_web_sm")
sentiment_model = pipeline("sentiment-analysis")
sentiment_vader = SentimentIntensityAnalyzer()

# Sample product list (extend as needed)
PRODUCT_KEYWORDS = ["Smartwatch X300", "Premium Plan", "Gold Subscription"]

# Empathy indicators
EMPATHY_KEYWORDS = ["I understand", "I'm sorry", "I apologize", "That must be frustrating"]

def load_transcript(file_path):
    ext = Path(file_path).suffix
    if ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_dict("records")
    elif ext == ".txt":
        with open(file_path, "r") as f:
            lines = f.readlines()
            result = []
            for line in lines:
                if ":" in line:
                    speaker, text = line.split(":", 1)
                    result.append({"Speaker": speaker.strip(), "Text": text.strip()})
            return result
    else:
        raise ValueError("Unsupported file format. Use .txt or .csv")

def extract_entities(text):
    doc = nlp_spacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def detect_products(text):
    return [product for product in PRODUCT_KEYWORDS if product.lower() in text.lower()]

def analyze_sentiment(text):
    score = sentiment_vader.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def detect_empathy(text):
    return any(phrase.lower() in text.lower() for phrase in EMPATHY_KEYWORDS)

def mock_intent(text):
    # Simplified intent classifier based on keywords
    if any(x in text.lower() for x in ["not working", "issue", "problem", "frustrated"]):
        return "Complaint"
    elif any(x in text.lower() for x in ["how", "can i", "what is"]):
        return "Inquiry"
    elif any(x in text.lower() for x in ["thanks", "appreciate"]):
        return "Appreciation"
    else:
        return "Other"

def analyze_transcript(conversation):
    output = []
    for turn in conversation:
        text = turn['Text']
        analysis = {
            "Speaker": turn['Speaker'],
            "Text": text,
            "Entities": extract_entities(text),
            "ProductMentions": detect_products(text),
            "Sentiment": analyze_sentiment(text),
            "Intent": mock_intent(text),
            "EmpathyDetected": detect_empathy(text) if turn['Speaker'].lower() == 'agent' else False
        }
        output.append(analysis)
    return output

def main(file_path):
    conversation = load_transcript(file_path)
    results = analyze_transcript(conversation)
    print(json.dumps(results, indent=2))

    # Save to JSON
    output_file = "output_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI/ML Interview Transcript Analysis")
    parser.add_argument("--file", type=str, required=True, help="Path to transcript file (.txt or .csv)")
    args = parser.parse_args()

    main(args.file)
