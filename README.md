
# Transcript Analysis with Gemini LLM

This branch implements an LLM-powered approach for analyzing customer-agent conversation transcripts using Google's Gemini Pro model. It enhances the capabilities of the original pipeline by leveraging generative AI for deeper analysis.

## Features

- Named Entity Recognition (NER): Identifies entities such as people, companies, and products.
- Product Mention Detection: Detects product references within customer queries.
- Sentiment Analysis: Labels each customer utterance as positive, negative, or neutral.
- Intent Recognition: Identifies the underlying intent of each customer message.
- Empathy Detection: Determines whether the agent expressed empathy, and extracts supporting examples.

## File Structure

- `main_llm.py`: Script for processing the transcript using the Gemini API.
- `convo.txt`: Sample input conversation in plain text format.
- `output_analysis_llm.json`: Output file with structured analysis results.
- `requirements.txt`: List of required Python packages.

## Setup Instructions

1. Clone the repository and switch to this branch:
   ```bash
   git checkout llm-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set your Gemini API key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
   Alternatively, you can directly assign the key in `main_llm.py`.

4. Run the script:
   ```bash
   python main_llm.py
   ```

## Example Output

```json
{
  "named_entities": [{"text": "John", "type": "Person"}],
  "product_mentions": ["Smart Speaker"],
  "sentiment_analysis": [{"utterance": "It's not working!", "sentiment": "negative"}],
  "intents": [{"utterance": "I need a replacement", "intent": "Request Replacement"}],
  "empathy": {
    "is_empathy_shown": true,
    "examples": ["I understand how frustrating this must be for you."]
  }
}
```

## Requirements

- Python 3.8 or higher
- `google-generativeai>=0.4.1`

## Notes

- Input must be a customer-agent transcript formatted as plain text.
- The Gemini API may return slight variations in output formatting. Basic error handling is included.
