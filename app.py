from flask import Flask, request, jsonify
from dotenv import load_dotenv
import joblib
import re
import html
import string
import numpy as np
import os

load_dotenv()

app = Flask(__name__, static_folder='public', static_url_path='')

api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    from google import genai
    client = genai.Client(api_key=api_key)

logreg_pipeline = None
if os.path.exists('logreg_pipeline.joblib'):
    logreg_pipeline = joblib.load('logreg_pipeline.joblib')

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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = EMOJI_PATTERN.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tweet = data.get('message', '')
    
    if not tweet:
        return jsonify({'error': 'No message provided'}), 400
    
    if logreg_pipeline is None:
        return jsonify({'error': 'Model not found'}), 500
    
    cleaned = clean_text(tweet)
    processed = cleaned.split()
    
    probs = logreg_pipeline.predict_proba([processed])[0]
    best_idx = np.argmax(probs)
    intent = logreg_pipeline.classes_[best_idx]
    confidence = float(probs[best_idx])
    
    handler = "Logistic Regression (Tier 1)"
    if confidence < 0.70 and client:
        handler = "Gemini 2.5 Flash (Tier 2)"
    
    reply = "Please DM us your order ID. ^AMZ"
    if client:
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f'Draft a short, polite Amazon support reply (1-2 sentences) ending with ^AMZ: {tweet}'
            )
            reply = response.text.strip()
        except:
            pass
    
    return jsonify({
        'intent': intent,
        'confidence': confidence,
        'handler': handler,
        'decision': 'ESCALATE_TO_HUMAN' if confidence < 0.70 else 'AUTO_HANDLE',
        'reply': reply
    })

if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv('PORT', 8080)))