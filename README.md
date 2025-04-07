
```markdown
# 🎧 Transcript Analysis using Pretrained NLP Models

This project analyzes customer-agent conversations using pre-trained NLP models. It extracts insights like named entities, product mentions, sentiment, intent, and empathy using state-of-the-art libraries.

---

## 🔍 Features

- **Named Entity Recognition (NER)**  
- **Product Mention Extraction** using KeyBERT  
- **Sentiment Analysis** using HuggingFace Transformers  
- **Intent Recognition**  
- **Empathy Detection** using keyword+semantic methods  
- **Keyword Summarization**

---

## 🧠 Models & Libraries Used

- `transformers`
- `sentence-transformers`
- `keybert`
- `scikit-learn`
- `spacy`
- `nltk`

---

## 📁 Folder Structure

```
transcript-analysis/
├── convo.txt               # Raw input transcript
├── main.py                 # Main pipeline script
├── output_analysis.json    # Output JSON with all analysis
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/vidhirana10/transcript-analysis.git
   cd transcript-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run analysis:
   ```bash
   python main.py
   ```

---

## 🛠 TODO

- Replace hardcoded product & empathy detection with LLM or semantic techniques
- Add web UI for interaction
- Better intent classification via fine-tuned model

---

## 💬 Sample Output

```json
{
  "Customer": {
    "sentiment": "neutral",
    "intent": "ask_issue",
    "empathy_detected": false,
    ...
  }
}
```

---

## 📌 Author

**Vidhi Rana**

---

Would you like to customize the title, TODOs, or output section? Let me know and I’ll help edit it before you commit and push it.
