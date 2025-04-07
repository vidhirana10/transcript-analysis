
# Transcript Analysis using Pretrained NLP Models

This project analyzes customer-agent call transcripts using state-of-the-art NLP techniques. It extracts key insights such as product mentions, named entities, customer intent, sentiment, and empathy detection — all using pretrained models, making it lightweight, easy to deploy, and language-flexible.

---

## Features

- **Named Entity Recognition (NER)**  
  Identifies people, organizations, dates, and other important entities in the conversation.

- **Product Mention Extraction**  
  Extracts products from conversation context using semantic keyword extraction.

- **Customer Sentiment Analysis**  
  Understands customer sentiment per utterance using transformer-based models.

- **Intent Detection**  
  Classifies customer intent like Complaint, Query, or Gratitude.

- **Empathy Detection**  
  Detects empathetic responses from the agent using pretrained embeddings and semantic matching.

---

## Setup

```bash
git clone https://github.com/vidhirana10/transcript-analysis.git
cd transcript-analysis
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## Input Format

Place your conversation in a file named `convo.txt` using the following format:

```
Customer: I’m having trouble with the GalaxyTab.
Agent: I’m sorry to hear that. Could you explain what’s going wrong?
Customer: The screen keeps flickering randomly.
```

---

## Output

Results will be saved in `output_analysis.json` with the following structure:

```json
[
  {
    "speaker": "Customer",
    "text": "I’m having trouble with the GalaxyTab.",
    "entities": ["GalaxyTab"],
    "intent": "Complaint",
    "sentiment": "negative"
  },
  ...
]
```

---

## TODOs

- Replace hardcoded product and empathy detection with semantic techniques or LLMs
- Add simple web UI for uploading files and viewing insights
- Improve intent classification via fine-tuned models

---

## Built With

- HuggingFace Transformers
- KeyBERT
- Sentence Transformers
- Python and JSON for processing

---

## License

MIT License

---

## Author

Vidhi Rana  
[GitHub](https://github.com/vidhirana10) | [LinkedIn](https://linkedin.com/in/vidhirana10)
