import os
import re
import string
import html
import numpy as np
import joblib
from google import genai
from dotenv import load_dotenv

load_dotenv()

def handler(event, context):
    """Vercel serverless handler for /api/analyze"""
    
    if event.get('method') != 'POST':
        return {'statusCode': 405, 'body': 'Method not allowed'}
    
    data = event.get('body', {})
    if isinstance(data, str):
        import json
        try:
            data = json.loads(data)
        except:
            return {'statusCode': 400, 'body': 'Invalid JSON'}
    
    tweet = data.get('message', '')
    if not tweet:
        return {'statusCode': 400, 'body': 'No message provided'}
    
    api_key = os.getenv('GEMINI_API_KEY', '')
    client = genai.Client(api_key=api_key) if api_key else None
    
    # Load model
    logreg_pipeline = None
    if os.path.exists('logreg_pipeline.joblib'):
        logreg_pipeline = joblib.load('logreg_pipeline.joblib')
    
    if logreg_pipeline is None:
        return {'statusCode': 500, 'body': 'Model not found'}
    
    # Preprocessing
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    
    cleaned = tweet
    cleaned = html.unescape(cleaned)
    cleaned = EMOJI_PATTERN.sub('', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lower()
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    processed = cleaned.split()
    
    # Predict
    probs = logreg_pipeline.predict_proba([processed])[0]
    best_idx = np.argmax(probs)
    intent = logreg_pipeline.classes_[best_idx]
    confidence = float(probs[best_idx])
    
    # Use LLM if confidence low
    if confidence < 0.70 and client:
        handler = "Gemini 2.5 Flash (Tier 2)"
    else:
        handler = "Logistic Regression (Tier 1)"
    
    # Generate reply
    reply = "Please DM us your order ID. ^AMZ"
    if client:
        try:
            llm_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Draft a short, polite Amazon support reply (max 1-2 sentences) ending with ^AMZ: {tweet}"
            )
            reply = llm_response.text.strip()
        except:
            pass
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': {
            'intent': intent,
            'confidence': confidence,
            'handler': handler,
            'decision': 'ESCALATE_TO_HUMAN' if confidence < 0.70 else 'AUTO_HANDLE',
            'reply': reply
        }
    }
