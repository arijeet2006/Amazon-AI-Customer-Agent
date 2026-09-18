<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&color=FF9900&center=true&vCenter=true&width=800&lines=Amazon+AI+Customer+Support+Triage+Agent;Two-Tier+Hybrid+ML+%2B+LLM+Routing+Engine;TF-IDF+%2B+Logistic+Regression+%E2%9A%A1;Gemini+2.5+Flash+Contextual+Fallback+%F0%9F%A7%A0;100%25+Effective+Accuracy+%E2%80%A2+Sub-5ms+Latency" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF%20%2B%20LogReg-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production--Grade-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**A production-grade, two-tier hybrid support routing engine 🎯 pairing sub-5ms classical ML with contextual LLM reasoning to hit 100% effective resolution accuracy.**

[Overview](#-architecture-overview) ‣ [Pipeline](#-notebook-walkthrough-modelipynb) ‣ [Benchmarks](#-benchmark-validation-classical-ml-vs-llm) ‣ [Intents](#-supported-intents) ‣ [Getting Started](#-getting-started)

</div>

---

## 🏛️ Architecture Overview

High-volume customer support operations face a fundamental trade-off between **cost/latency** and **contextual comprehension**. This project resolves that trade-off with a **cascaded two-tier routing architecture**: a fast, cheap classical model handles the bulk of clear-cut traffic, and a large language model is reserved only for the genuinely hard cases.

```mermaid
flowchart TD
    A["📥 Incoming Customer Tweet / Inquiry"] --> B["🔄 Text Preprocessing Pipeline<br/>HTML unescape → emoji strip → stemming"]
    B --> C["⚙️ Tier 1: TF-IDF + Logistic Regression<br/>Predicts Intent & Confidence"]
    C -->|"Confidence ≥ 0.70<br/>Clear, standard query"| D["⚡ Fast-Path Execution<br/>Local Inference → less than 5ms → $0 API cost"]
    C -->|"Confidence < 0.70<br/>Ambiguous, frustrated, edge case"| E["🧠 Contextual LLM Fallback<br/>Gemini 2.5 Flash"]
    E --> F["🔍 Deep Semantic Classification<br/>+ Escalation Detection"]
    F --> G["📝 Policy-Grounded Draft Reply<br/>Brand Voice: ^AMZ"]
    D --> H["✅ Resolved Ticket"]
    G --> H
```

| Tier | Engine | Trigger | Latency | Cost | Best For |
|------|--------|---------|---------|------|----------|
| **1 – Fast Path** | TF-IDF + Logistic Regression | Confidence ≥ 0.70 | < 5ms | $0 (no API call) | Repeated, direct, unambiguous queries |
| **2 – Contextual Fallback** | Gemini 2.5 Flash | Confidence < 0.70 | ~1–2s | Per-token API cost | Sarcasm, multi-intent, implicit or emotionally charged messages |

---

## 📓 Notebook Walkthrough (`model.ipynb`)

The full pipeline is built step-by-step inside `model.ipynb`.

### 1️⃣ Data Cleaning & Extraction
- Filters raw Twitter customer service data (`twcs.csv`) down to inbound customer tweets paired with official **AmazonHelp** responses.
- Enforces strict Latin-character validation to filter out non-English interactions.
- Applies a regex-based emoji stripper and an HTML entity unescaper.

### 2️⃣ Preprocessing Pipeline
- Lowercasing, punctuation stripping, and tokenization via **NLTK**.
- English stopword pruning.
- Vocabulary reduction via the **Porter Stemmer** algorithm.

### 3️⃣ Model Training & Intent Labeling
A sublinear TF-IDF vectorizer (unigrams + bigrams, up to **10,000 features**) is paired with a **class-balanced Logistic Regression** solver, trained across six operational intents.

### 4️⃣ Benchmark Validation
The hybrid system is stress-tested against a held-out set of real-world, ambiguous customer inquiries — see results below.

---

## 🎯 Supported Intents

<div align="center">

| Intent | Description |
|---|---|
| 🚚 `order_tracking` | Delivery delays, missing parcels, transit inquiries |
| 💸 `refund_cancellation` | Return policy, subscription cancellations, refund requests |
| 📦 `damaged_defective` | Broken items, opened seals, incorrect products received |
| 🔐 `account_access` | Two-factor authentication, login errors, password resets |
| 💳 `billing_charge` | Unrecognized charges, double billing, card disputes |
| 💬 `general_inquiry` | General feedback, partnership inquiries, generic support |

</div>

---

## 📊 Benchmark Validation: Classical ML vs. LLM

Tested against an evaluation set of real-world, ambiguous customer inquiries:

<div align="center">

| Approach | Accuracy | Notes |
|---|:---:|---|
| Logistic Regression alone | **60.0%** | Fails on implicit intent — e.g. classifies *"Your driver threw the box over my fence in the pouring rain"* as `general_inquiry` instead of `order_tracking` |
| Gemini 2.5 Flash alone | **100.0%** | Correctly resolves sarcasm, unstated intent, and implicit problems |
| **Hybrid Two-Tier System** | **100.0%** (effective) | Routes the majority of clear-cut queries locally, escalating only ambiguous cases to the LLM |

</div>

> **Key takeaway:** the hybrid architecture matches full-LLM accuracy while dramatically cutting inference cost and latency, since only low-confidence tickets ever reach the Gemini fallback.

---

## 🏗️ Why a Two-Tier System?

- ⚡ **Speed at scale** — the majority of support volume is repetitive and unambiguous; resolving it in under 5ms avoids unnecessary LLM round-trips.
- 💰 **Cost efficiency** — every ticket resolved locally is a ticket that costs $0 in API tokens.
- 🎯 **No accuracy trade-off** — the 0.70 confidence threshold acts as a safety net, ensuring anything the classical model isn't sure about still gets full contextual reasoning.
- 🔄 **Graceful escalation** — Gemini 2.5 Flash doesn't just classify; it determines whether a human agent needs to step in and drafts a policy-grounded, on-brand reply.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/amazon-support-triage-agent.git
cd amazon-support-triage-agent

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook model.ipynb
```

### Environment Variables

```bash
GEMINI_API_KEY=your_google_gemini_api_key
```

---

## 🏗️ Project Structure

```text
amazon-support-triage-agent/
├── model.ipynb          # Full pipeline: data cleaning → training → benchmarking
├── twcs.csv             # Raw Twitter customer support dataset
├── requirements.txt     # Python dependencies
├── README.md
```

---

## 🚀 Deployment Options

### Best Options for Streamlit

| Platform | Cost | Ease | Recommendation |
|----------|------|------|----------------|
| **Streamlit Cloud** | Free | ⭐⭐⭐⭐⭐ | ✅ Best choice |
| **Hugging Face Spaces** | Free | ⭐⭐⭐⭐ | ✅ Good alternative |
| **Vercel** | Free tier | ⭐⭐ | ⚠️ Not ideal for Streamlit |

### Deploy to Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo and deploy

### Deploy to Hugging Face Spaces
1. Create a new Space at https://huggingface.co/spaces
2. Select "Streamlit" as the SDK
3. Upload your `app.py`, `requirements.txt`, and `logreg_pipeline.joblib`

### Vercel (Not Recommended)
Vercel's serverless architecture is not well-suited for Streamlit's persistent server model. For API-based deployments, consider migrating to FastAPI or Flask.

---

<div align="center">

Set GEMINI_API_KEY in Vercel dashboard before deploying.

