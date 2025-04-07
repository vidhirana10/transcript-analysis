import google.generativeai as genai
import json

# Configure your Gemini API key
genai.configure(api_key="YOUR-API-KEY")  # Replace with your actual API key

def analyze_with_gemini(transcript_text):
    prompt = """
You are an intelligent assistant analyzing a conversation between a customer and an agent. Your tasks are:

1. Named Entity Recognition (NER): Identify all named entities and their types.
2. Product Mentions: List all products mentioned.
3. Sentiment Analysis: Analyze the sentiment of the customer per utterance (positive, negative, neutral).
4. Intent Recognition: Identify the intent behind customer utterances.
5. Empathy Detection: Determine if the agent shows empathy and provide examples.

Return the output strictly in the following JSON format:
{
  "named_entities": [
    {"text": "...", "type": "..."}
  ],
  "product_mentions": [
    "..."
  ],
  "sentiment_analysis": [
    {"utterance": "...", "sentiment": "..."}
  ],
  "intents": [
    {"utterance": "...", "intent": "..."}
  ],
  "empathy": {
    "is_empathy_shown": true,
    "examples": ["..."]
  }
}

Transcript:
""" + transcript_text

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)

    try:
        response_text = response.text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[-1].split("```")[0].strip()

        return json.loads(response_text)
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Full response:\n", response.text)
        return {}

if __name__ == "__main__":
    try:
        with open("convo.txt", "r", encoding="utf-8") as f:
            transcript = f.read()

        results = analyze_with_gemini(transcript)

        with open("output_analysis_llm.json", "w", encoding="utf-8") as out:
            json.dump(results, out, indent=4)
        
        print("Analysis complete. Results saved to output_analysis.json")
    
    except FileNotFoundError:
        print("'convo.txt' not found. Please make sure the file exists in the same directory.")
