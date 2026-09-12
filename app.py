import html
import os
import re
import string
from dotenv import load_dotenv
from google import genai
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Amazon Support Triage Agent",
    page_icon="📦",
    layout="centered",
)

# Minimal CSS (Theme & Custom Badges)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
    }
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #f3f4f6 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px !important;
    }
    .badge-auto {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .badge-escalate {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 5px 12px;
        border-radius: 6px;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: inline-block;
    }
    .reply-card {
        background-color: rgba(245, 158, 11, 0.05);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 16px;
        margin-top: 14px;
        font-style: italic;
        color: #f9fafb;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# 2. Environment & Gemini Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

# 3. Exact Preprocessing from Notebook (Cell 66)
nltk_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_dir, quiet=True)
nltk.download("punkt_tab", download_dir=nltk_dir, quiet=True)
nltk.download("stopwords", download_dir=nltk_dir, quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

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


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = EMOJI_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_pipeline(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return [stemmer.stem(w) for w in tokens]


# 4. Load Pipeline Artifact (Cell 81 & Notebook save)
@st.cache_resource
def load_pipeline():
    if os.path.exists("logreg_pipeline.joblib"):
        return joblib.load("logreg_pipeline.joblib")
    return None


logreg_pipeline = load_pipeline()


# 5. Hybrid Classification & Agent from Notebook (Cells 81 & 83)
def classify_customer_tweet(tweet_text: str, confidence_cutoff: float = 0.70):
    cleaned = clean_text(tweet_text)
    processed = " ".join(preprocess_pipeline(cleaned))

    probs = logreg_pipeline.predict_proba([processed])[0]
    best_idx = np.argmax(probs)
    logreg_intent = logreg_pipeline.classes_[best_idx]
    confidence = probs[best_idx]

    if confidence >= confidence_cutoff:
        return {
            "tweet": tweet_text,
            "handler": "Logistic Regression (Tier 1)",
            "intent": logreg_intent,
            "confidence": round(float(confidence), 2),
        }

    llm_prompt = f"""
    Classify this Amazon customer tweet into exactly one intent:
    (order_tracking, refund_cancellation, damaged_defective, account_access, billing_charge, general_inquiry)

    Tweet: "{tweet_text}"
    Return ONLY the category name.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=llm_prompt
    )
    llm_intent = response.text.strip().lower()

    return {
        "tweet": tweet_text,
        "handler": "Gemini 2.5 Flash (Tier 2 Fallback)",
        "intent": llm_intent,
        "confidence": round(float(confidence), 2),
    }


def amazon_agent(tweet: str):
    prompt = f"""
    You are an Amazon Twitter support agent.
    Analyze this customer tweet: "{tweet}"

    Give your answer in EXACTLY this format (do not add extra text):
    Intent: (choose one: order_tracking, refund_cancellation, damaged_defective, account_access, billing_charge, general_inquiry)
    Decision: (choose one: AUTO_HANDLE or ESCALATE_TO_HUMAN)
    Reason: (one short sentence why)
    Reply: (short polite Amazon tweet reply ending with ^AMZ)
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text


# 6. Streamlit User Interface
st.title("📦 Amazon Support Triage Agent")
st.caption("Classical ML (Tier 1) + Gemini 2.5 Flash (Tier 2 Contextual)")

# Quick Preset Buttons from Notebook Test Queries (Cells 82 & 84)
col1, col2 = st.columns(2)
preset = None
if col1.button("📍 Missing Package"):
    preset = "Where is my package? It was supposed to be delivered yesterday."
if col2.button("⚠️ Wrong Item Delivered"):
    preset = "I ordered a blue sweater and you guys sent me a coffee mug instead???"

user_tweet = st.text_area(
    "Customer Message",
    value=preset or "",
    placeholder="Type or select a customer query above...",
    height=90,
    label_visibility="collapsed",
)

if st.button(
    "Analyze & Draft Response", type="primary", use_container_width=True
):
    if not user_tweet.strip():
        st.warning("Please enter a customer message.")
    elif logreg_pipeline is None:
        st.error("Missing `logreg_pipeline.joblib` in the root folder.")
    elif not client:
        st.error("Missing `GEMINI_API_KEY` in environment / .env file.")
    else:
        with st.spinner("Processing triage..."):
            # Step 1: Hybrid Classification (Cell 81)
            clf_result = classify_customer_tweet(user_tweet)

            # Step 2: Agent Response Generation (Cell 83)
            agent_raw = amazon_agent(user_tweet)

            # Parse agent key-value pairs
            agent_data = {}
            for line in agent_raw.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    agent_data[k.strip().lower()] = v.strip()

            decision = agent_data.get("decision", "AUTO_HANDLE")
            reason = agent_data.get("reason", "Standard operational handling.")
            reply = agent_data.get(
                "reply", "Please DM us your order ID so we can assist. ^AMZ"
            )

            st.markdown("---")

            # Clean UI Presentation
            top_c1, top_c2 = st.columns([1, 1])
            with top_c1:
                st.caption("TRIAGE ACTION")
                badge_style = (
                    "badge-escalate" if "ESCALATE" in decision else "badge-auto"
                )
                st.markdown(
                    f'<span class="{badge_style}">{decision}</span>',
                    unsafe_allow_html=True,
                )

            with top_c2:
                st.caption("ROUTED BY")
                st.markdown(f"**{clf_result['handler']}**")

            st.markdown(f"**Routing Basis:** {reason}")

            st.caption("BRAND-GROUNDED DRAFT REPLY")
            st.markdown(
                f'<div class="reply-card">{reply}</div>', unsafe_allow_html=True
            )