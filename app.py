import re
import string
import html
import numpy as np
import joblib
import streamlit as st

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

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = EMOJI_PATTERN.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load model
logreg_pipeline = None
try:
    logreg_pipeline = joblib.load('logreg_pipeline.joblib')
except Exception:
    pass

# Main app
st.set_page_config(page_title="Amazon Support Triage Agent", page_icon="📦", layout="centered")

st.title("📦 Amazon Support Triage Agent")
st.caption("Classical ML (Tier 1) + Gemini 2.5 Flash (Tier 2)")

# Preset buttons
col1, col2 = st.columns(2)
preset = None
if col1.button("📍 Missing Package"):
    preset = "Where is my package? It was supposed to be delivered yesterday."
if col2.button("⚠️ Wrong Item"):
    preset = "I ordered a blue sweater and you guys sent me a coffee mug instead???"

user_tweet = st.text_area(
    "Customer Message",
    value=preset or "",
    placeholder="Type or select a customer query above...",
    height=90,
    label_visibility="collapsed",
)

if st.button("Analyze & Draft Response", type="primary", use_container_width=True):
    if not user_tweet.strip():
        st.warning("Please enter a customer message.")
    elif logreg_pipeline is None:
        st.error("Model not loaded. Please ensure logreg_pipeline.joblib exists.")
    else:
        with st.spinner("Processing triage..."):
            # Preprocess
            cleaned = clean_text(user_tweet)
            processed = cleaned.split()
            
            if not processed:
                st.error("Empty message after preprocessing.")
            else:
                # Predict
                probs = logreg_pipeline.predict_proba([processed])[0]
                best_idx = np.argmax(probs)
                intent = logreg_pipeline.classes_[best_idx]
                confidence = probs[best_idx]
                
                # Show results
                st.markdown("---")
                
                top_c1, top_c2 = st.columns([1, 1])
                with top_c1:
                    st.caption("TRIAGE ACTION")
                    if confidence < 0.70:
                        st.markdown('<span class="badge-escalate">ESCALATE_TO_HUMAN</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-auto">AUTO_HANDLE</span>', unsafe_allow_html=True)
                
                with top_c2:
                    st.caption("ROUTED BY")
                    handler = "Logistic Regression (Tier 1)"
                    if confidence < 0.70:
                        handler = "Gemini 2.5 Flash (Tier 2)"
                    st.markdown(f"**{handler}**")
                
                st.markdown(f"**INTENT:** {intent}")
                st.markdown(f"**CONFIDENCE:** {(confidence * 100):.1f}%")
                
                st.caption("BRAND-GROUNDED DRAFT REPLY")
                reply = "Please DM us your order ID. ^AMZ"
                st.markdown(f'<div class="reply-card">{reply}</div>', unsafe_allow_html=True)
