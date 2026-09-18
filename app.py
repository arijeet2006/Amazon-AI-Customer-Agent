import os
import re
import string
import html
import json
import numpy as np
import joblib
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Load model at startup
logreg_pipeline = None
if os.path.exists('logreg_pipeline.joblib'):
    logreg_pipeline = joblib.load('logreg_pipeline.joblib')

# Setup Gemini client
api_key = os.getenv('GEMINI_API_KEY', '')
client = None
if api_key:
    client = genai.Client(api_key=api_key)

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

def handler(request):
    """Vercel serverless handler - always returns JSON"""
    try:
        method = request.get('method', '').upper()
        
        if method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Content-Type': 'application/json'
                },
                'body': ''
            }
        
        if method == 'POST':
            # Parse request body
            body = request.get('body', '')
            data = {}
            if body:
                try:
                    data = json.loads(body)
                except:
                    data = {}
            
            tweet = str(data.get('message', ''))
            
            if not tweet:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'intent': 'general_inquiry',
                        'confidence': 0.0,
                        'handler': 'Error',
                        'decision': 'ERROR',
                        'reply': 'No message provided.'
                    })
                }
            
            if logreg_pipeline is None:
                return {
                    'statusCode': 500,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'intent': 'general_inquiry',
                        'confidence': 0.0,
                        'handler': 'Error',
                        'decision': 'ERROR',
                        'reply': 'Model not loaded.'
                    })
                }
            
            cleaned = clean_text(tweet)
            processed = cleaned.split()
            
            if not processed:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'intent': 'general_inquiry',
                        'confidence': 0.0,
                        'handler': 'Error',
                        'decision': 'ERROR',
                        'reply': 'Empty message.'
                    })
                }
            
            probs = logreg_pipeline.predict_proba([processed])[0]
            best_idx = int(np.argmax(probs))
            intent = str(logreg_pipeline.classes_[best_idx])
            confidence = float(probs[best_idx])
            
            handler_name = "Logistic Regression (Tier 1)"
            if confidence < 0.70 and client:
                handler_name = "Gemini 2.5 Flash (Tier 2)"
            
            reply = "Please DM us your order ID. ^AMZ"
            if client:
                try:
                    llm_response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents='Reply: ' + tweet
                    )
                    if llm_response and llm_response.text:
                        reply = str(llm_response.text).strip()
                except:
                    pass
            
            decision = 'ESCALATE_TO_HUMAN' if confidence < 0.70 else 'AUTO_HANDLE'
            
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'intent': intent,
                    'confidence': confidence,
                    'handler': handler_name,
                    'decision': decision,
                    'reply': reply
                })
            }
        
        # Default response for any other request
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'message': 'Amazon Support Triage Agent API',
                'status': 'ok'
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'intent': 'general_inquiry',
                'confidence': 0.0,
                'handler': 'Error',
                'decision': 'ERROR',
                'reply': 'An error occurred: ' + str(e)
            })
        }
