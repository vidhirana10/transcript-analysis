import os
import json
import pandas as pd
import spacy
from transformers import pipeline
from pathlib import Path
from keybert import KeyBERT

# Load models
nlp_spacy = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

# === Replacing VADER with Emotion Transformer ===
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

empathy_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# === File Loader ===
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

# === Entity Recognition ===
def extract_entities(text):
    doc = nlp_spacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# === Dynamic Product Mention Detection ===
def detect_products(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    return [kw[0] for kw in keywords]

# === Emotion-Based Sentiment Analysis ===
def analyze_sentiment(text):
    result = emotion_classifier(text)
    return result[0]['label'].capitalize()

# === Empathy Detection using Zero-shot Classification ===
def detect_empathy(text):
    candidate_labels = ["empathy", "no empathy"]
    result = empathy_classifier(text, candidate_labels)
    return result['labels'][0] == "empathy"

# === Intent Detection (Mock) ===
def mock_intent(text):
    text_lower = text.lower()
    if any(x in text_lower for x in ["not working", "issue", "problem", "frustrated"]):
        return "Complaint"
    elif any(x in text_lower for x in ["how", "can i", "what is"]):
        return "Inquiry"
    elif any(x in text_lower for x in ["thanks", "appreciate"]):
        return "Appreciation"
    else:
        return "Other"

# === Analyzer ===
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

# === Main Entry ===
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
