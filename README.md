
# Transcript Analysis using Pretrained NLP Models

This project analyzes customer-agent call transcripts using state-of-the-art NLP techniques. It extracts key insights such as product mentions, named entities, customer intent, emotion-based sentiment, and empathy detection — all using pretrained models, making it lightweight, scalable, and ready for real-world applications.

---

## 🔍 Features

- **Named Entity Recognition (NER)**  
  Identifies names, dates, locations, and other key entities using spaCy.

- **Product Mention Extraction**  
  Extracts relevant products dynamically using `KeyBERT` with contextual embedding.

- **Emotion-Based Sentiment Analysis**  
  Uses a RoBERTa-based transformer model to detect nuanced emotions like _joy, sadness, anger, love, optimism,_ and _fear_.

- **Intent Detection (Rule-based)**  
  Classifies user intent into categories such as _Complaint_, _Inquiry_, or _Appreciation_.

- **Empathy Detection (Zero-Shot)**  
  Uses zero-shot classification with BART to detect empathetic behavior in agent responses.

---

## ⚙️ Setup

```bash
git clone https://github.com/vidhirana10/transcript-analysis.git
cd transcript-analysis
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python main.py --file convo.txt
```

---

## 📝 Input Format

Create a transcript file named `convo.txt` in the following format:

```
Customer: I'm facing issues with my SmartWatch Pro.
Agent: I'm really sorry to hear that! Can you describe the issue?
Customer: The screen keeps freezing and it's very frustrating.
```

---

## 📤 Output

After analysis, results will be saved in `output_analysis.json` with a structure like:

```json
[
  {
    "Speaker": "Customer",
    "Text": "I'm facing issues with my SmartWatch Pro.",
    "Entities": [["SmartWatch Pro", "PRODUCT"]],
    "ProductMentions": ["smartwatch"],
    "Sentiment": "anger",
    "Intent": "Complaint",
    "EmpathyDetected": false
  },
  ...
]
```

---

## 🛠 TODOs

- Add UI for drag-and-drop conversation uploads
- Integrate Gemini/GPT-based LLM pipeline
- Replace rule-based intent detection with fine-tuned classifier
- Visualize sentiment/empathy trends over time

---

## 📦 Built With

- **spaCy** for NER  
- **KeyBERT** for keyword extraction  
- **RoBERTa** for emotion sentiment classification (`j-hartmann/emotion-english-distilroberta-base`)  
- **BART (zero-shot)** for empathy classification  
- **Python** + **JSON** for lightweight processing

---

## 📄 License

MIT License

---

## 👩🏻‍💻 Author

**Vidhi Rana**  
