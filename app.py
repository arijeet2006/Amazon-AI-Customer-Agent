import os
import re
import string
import html
import numpy as np
import joblib
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

load_dotenv()

app = Flask(__name__)

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
    return send_file('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    tweet = ''
    try:
        data = request.get_json(force=True, silent=True)
        if data:
            tweet = str(data.get('message', ''))
    except:
        pass
    
    if not tweet:
        return jsonify({'intent': 'general_inquiry', 'confidence': 0.0, 'handler': 'Error', 'decision': 'ERROR', 'reply': 'Please provide a message.'})
    
    if logreg_pipeline is None:
        return jsonify({'intent': 'general_inquiry', 'confidence': 0.0, 'handler': 'Model Not Loaded', 'decision': 'ERROR', 'reply': 'Model not found.'})
    
    try:
        cleaned = clean_text(tweet)
        processed = cleaned.split()
        
        if not processed:
            return jsonify({'intent': 'general_inquiry', 'confidence': 0.5, 'handler': 'Default', 'decision': 'AUTO_HANDLE', 'reply': 'Please provide a valid message. ^AMZ'})
        
        probs = logreg_pipeline.predict_proba([processed])[0]
        best_idx = int(np.argmax(probs))
        intent = str(logreg_pipeline.classes_[best_idx])
        confidence = float(probs[best_idx])
        
        handler = "Logistic Regression (Tier 1)"
        if confidence < 0.70 and client:
            handler = "Gemini 2.5 Flash (Tier 2)"
        
        reply = "Please DM us your order ID. ^AMZ"
        if client:
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents='Draft a short Amazon support reply (max 2 sentences) ending with ^AMZ: ' + tweet
                )
                if response and hasattr(response, 'text') and response.text:
                    reply = str(response.text).strip()
            except Exception:
                pass
        
        decision = 'ESCALATE_TO_HUMAN' if confidence < 0.70 else 'AUTO_HANDLE'
        
        return jsonify({
            'intent': intent,
            'confidence': confidence,
            'handler': handler,
            'decision': decision,
            'reply': reply
        })
    except Exception as e:
        return jsonify({'intent': 'general_inquiry', 'confidence': 0.0, 'handler': 'Error', 'decision': 'ERROR', 'reply': 'An error occurred. Please try again.'})
