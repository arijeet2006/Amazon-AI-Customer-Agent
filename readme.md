# 📦 Amazon AI Customer Support Triage Agent

A production-grade, two-tier hybrid customer support routing and response generation system. The engine combines high-speed, local classical machine learning (TF-IDF + Logistic Regression) with contextual Large Language Model reasoning (Google Gemini 2.5 Flash) to balance sub-millisecond execution speeds with 100% resolution accuracy on complex edge cases.

---

## 🎯 Architecture Overview: Two-Tier Hybrid Routing

High-volume customer support operations typically face a trade-off between **cost/latency** and **contextual comprehension**. This project implements a cascaded routing architecture:

```text
                  Incoming Customer Tweet / Inquiry
                                  │
                                  ▼
                 ┌──────────────────────────────────┐
                 │    Text Preprocessing Pipeline   │
                 │ (HTML unescape, emojis, stemming)│
                 └──────────────────────────────────┘
                                  │
                                  ▼
                 ┌──────────────────────────────────┐
                 │  Tier 1: TF-IDF + Logistic Reg   │
                 │   Predicts Intent & Confidence   │
                 └──────────────────────────────────┘
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
    Confidence ≥ 0.70                      Confidence < 0.70
   (Clear, standard query)                (Ambiguous, frustrated, edge-case)
               │                                     │
               ▼                                     ▼
     ⚡ Fast-Path Execution               🧠 Contextual LLM Fallback
    Local Machine Learning                    Google Gemini 2.5 Flash
   • Latency: < 5ms                          • Context-aware triage & escalation
   • Zero API token cost                     • Brand-grounded draft reply (^AMZ)


   Tier 1 (Classical Machine Learning): Evaluates incoming tweets using a TF-IDF vectorizer and Logistic Regression pipeline. Standard, repeated inquiries (e.g., direct tracking questions) achieve high confidence (≥ 0.70) and are resolved locally in milliseconds with zero LLM API costs.

   Tier 2 (Gemini 2.5 Flash Fallback): When the classical model encounters ambiguous phrasing, multi-intent queries, angry disputes, or edge cases (confidence < 0.70), the ticket escalates to Gemini 2.5 Flash. The LLM performs deep semantic classification, determines human escalation needs, and drafts a policy-grounded Amazon reply.

   🔬 Notebook Walkthrough (model.ipynb)

The system pipeline is constructed step-by-step in model.ipynb:
1. Data Cleaning & Extraction

    Filters raw Twitter customer service data (twcs.csv) down to inbound customer tweets and official AmazonHelp interactions.

    Enforces strictly Latin character validation to filter non-English interactions.

    Applies a regex-based emoji stripper and HTML entity unescaper.

2. Preprocessing Pipeline

    Lowercasing, punctuation stripping, and tokenization via NLTK.

    English stopword pruning.

    Vocabulary reduction via the Porter Stemmer algorithm.

3. Model Training & Intent Labeling

Customer messages are categorized across 6 operational intents:

    order_tracking: Delivery delays, missing parcels, transit inquiries.

    refund_cancellation: Return policy, subscription cancellations, refund requests.

    damaged_defective: Broken items, opened seals, incorrect products received.

    account_access: Two-factor authentication, login errors, password resets.

    billing_charge: Unrecognized charges, double billing, card disputes.

    general_inquiry: General customer feedback, partnership inquiries, generic support.

The pipeline applies a sublinear TF-IDF vectorizer (unigrams and bigrams, up to 10,000 features) coupled with a balanced LogisticRegression solver.


4. Benchmark Validation: Classical ML vs. LLM

When tested against an evaluation set of real-world ambiguous customer inquiries:

    Logistic Regression Alone: Achieves 60.0% accuracy due to keyword limitations (e.g., classifying "Your driver threw the box over my fence in the pouring rain" as a general inquiry instead of order tracking).

    Gemini 2.5 Flash: Achieves 100.0% accuracy by resolving sarcasm, unstated intentions, and implicit customer problems.

    Hybrid Two-Tier Setup: Retains 100% effective accuracy while routing the majority of clear-cut queries locally without API overhead.
