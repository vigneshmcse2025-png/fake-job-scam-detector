
# ============================================================
#  ScamShield – Fake Job Scam Detector
#  backend/app.py  |  Flask API Server
#
#  Endpoints:
#    POST /predict  – Accepts job details JSON, returns risk report
#    GET  /metrics  – Returns trained model comparison metrics
#    GET  /health   – Simple health check
# ============================================================

import json
import re
import os
import string
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App Setup ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow frontend (different port) to call this API

# ── Paths ────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH   = os.path.join(BASE_DIR, "metrics.json")

# ── Load Model & Vectorizer ──────────────────────────────
try:
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("[✓] Model and vectorizer loaded successfully.")
except FileNotFoundError:
    model      = None
    vectorizer = None
    print("[!] WARNING: model.pkl or vectorizer.pkl not found.")
    print("    Run:  python backend/train.py   to train and save the model first.")

# ── NLP Preprocessing Helpers ────────────────────────────
# These must match exactly what was done during training (train.py)

# Common English stopwords (lightweight – no NLTK dependency)
STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they",
    "what","which","who","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","do","does","did",
    "will","would","shall","should","may","might","must","can","could",
    "a","an","the","and","but","if","or","because","as","until","while",
    "of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","from","up","down",
    "in","out","on","off","over","under","again","then","once","here",
    "there","when","where","why","how","all","both","each","more","other",
    "some","such","no","nor","not","only","same","so","than","too","very",
    "just","don","doesn","didn","s","t","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn",
    "ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won",
    "wouldn"
}

def clean_text(text: str) -> str:
    """
    Basic NLP cleaning:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation & digits
    5. Remove extra spaces
    6. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove punctuation and digits
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def build_combined_text(data: dict) -> str:
    """
    Combine all relevant job fields into one text string.
    Order: title + company + description + salary + location + experience
    (Must match what was done in train.py)
    """
    parts = [
        data.get("job_title", ""),
        data.get("company", ""),
        data.get("description", ""),
        data.get("salary", ""),
        data.get("location", ""),
        data.get("experience", ""),
    ]
    return " ".join(str(p) for p in parts if p)


def compute_heuristic_warnings(data: dict, fraud_prob: float) -> tuple:
    """
    Rule-based heuristic checks on the raw job text.
    Returns:
        warnings    – list of human-readable warning strings
        risk_factors – list of {label, score} dicts
    """
    warnings     = []
    risk_factors = []
    desc         = data.get("description", "").lower()
    title        = data.get("job_title", "").lower()
    salary       = data.get("salary", "").lower()
    url          = data.get("job_url", "").lower()
    company      = data.get("company", "").lower()

    # ─ Description length
    word_count = len(desc.split())
    if word_count < 30:
        warnings.append("Job description is extremely short – genuine postings usually provide detailed information.")
        risk_factors.append({"label": "Description Length", "score": 0.85})
    elif word_count < 80:
        warnings.append("Job description is quite brief. Scam postings often lack detail.")
        risk_factors.append({"label": "Description Length", "score": 0.55})
    else:
        risk_factors.append({"label": "Description Length", "score": 0.1})

    # ─ Salary red flags
    salary_scam_keywords = ["work from home earn", "earn per day", "earn daily", "no experience needed",
                             "earn lakhs", "earn crores", "100000 per day", "unlimited earning"]
    for kw in salary_scam_keywords:
        if kw in salary or kw in desc:
            warnings.append(f"Suspicious earning claim detected: '{kw}'.")
            risk_factors.append({"label": "Salary Claims", "score": 0.9})
            break
    else:
        if not salary:
            risk_factors.append({"label": "Salary Disclosed", "score": 0.35})
        else:
            risk_factors.append({"label": "Salary Claims", "score": 0.15})

    # ─ Registration / upfront fee
    fee_keywords = ["registration fee", "deposit", "pay to apply", "training fee", "security deposit",
                     "refundable fee", "joining fee"]
    fee_found = any(kw in desc for kw in fee_keywords)
    if fee_found:
        warnings.append("Posting mentions fees/deposits – legitimate jobs NEVER ask candidates to pay.")
        risk_factors.append({"label": "Fee/Deposit Mention", "score": 0.95})
    else:
        risk_factors.append({"label": "Fee/Deposit Mention", "score": 0.05})

    # ─ Personal info harvesting
    pii_keywords = ["aadhar", "aadhaar", "pan number", "passport number", "bank account",
                     "share your id", "send your photo"]
    pii_found = any(kw in desc for kw in pii_keywords)
    if pii_found:
        warnings.append("Posting requests sensitive personal documents upfront – a major red flag.")
        risk_factors.append({"label": "PII Harvesting", "score": 0.9})
    else:
        risk_factors.append({"label": "PII Harvesting", "score": 0.05})

    # ─ Generic/vague title
    vague_titles = ["data entry", "home based", "work from home", "part time job",
                     "online job", "typing job", "form filling"]
    title_vague = any(v in title for v in vague_titles)
    if title_vague:
        warnings.append("Job title matches common scam patterns (data entry, typing, form filling).")
        risk_factors.append({"label": "Title Suspicion", "score": 0.75})
    else:
        risk_factors.append({"label": "Title Suspicion", "score": 0.15})

    # ─ URL quality
    suspicious_url_patterns = ["bit.ly", "tinyurl", "t.co", "goo.gl", "shorturl", "whatsapp", "telegram"]
    if any(p in url for p in suspicious_url_patterns):
        warnings.append("Job URL uses a URL shortener or messaging platform – common in scam listings.")
        risk_factors.append({"label": "Suspicious URL", "score": 0.8})
    elif not url:
        risk_factors.append({"label": "Job URL Present", "score": 0.25})
    else:
        risk_factors.append({"label": "Suspicious URL", "score": 0.1})

    # ─ ML model probability contributes too
    risk_factors.append({"label": "ML Model Score", "score": round(fraud_prob, 2)})

    # Sort by score descending
    risk_factors.sort(key=lambda x: x["score"], reverse=True)

    return warnings, risk_factors[:6]   # Return top-6 factors


def classify_risk(prob: float) -> str:
    """Map fraud probability to a risk level string."""
    if prob >= 0.75: return "Very High"
    if prob >= 0.50: return "High"
    if prob >= 0.30: return "Medium"
    return "Low"


def generate_recommendation(prediction: str, prob: float) -> str:
    """Generate a context-sensitive recommendation message."""
    if prediction == "fake":
        return ("❌ DO NOT apply to this job. This posting has a high probability of being fraudulent. "
                "Report it to the platform where you found it and warn others.")
    elif prediction == "suspicious":
        return ("⚠️ Exercise extreme caution. Before applying: (1) Verify the company on LinkedIn/official website, "
                "(2) Never pay any fee, (3) Do not share sensitive personal documents, "
                "(4) Search for the company's official HR contact independently.")
    else:
        return ("✅ This posting appears genuine, but always stay vigilant. Verify the company independently, "
                "never pay fees to apply, and be cautious about sharing personal documents.")


# ── /health ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({
        "status":        "ok",
        "model_loaded":  model is not None,
        "vectorizer_loaded": vectorizer is not None,
    })


# ── /predict ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Expects JSON body:
        {job_title, company, description, salary, location, experience, job_url}
    Returns JSON:
        {prediction, fraud_probability, confidence, risk_level,
         warnings, risk_factors, recommendation, model_used}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON payload."}), 400

    # Validate required fields
    if not data.get("description", "").strip():
        return jsonify({"error": "Job description is required."}), 400

    # ── Model prediction ──
    if model is None or vectorizer is None:
        # Fallback: heuristic-only mode when model hasn't been trained yet
        fraud_prob  = 0.5
        confidence  = 0.5
        model_name  = "Heuristic (model not trained)"
    else:
        combined_text = build_combined_text(data)
        cleaned_text  = clean_text(combined_text)
        vec           = vectorizer.transform([cleaned_text])
        proba         = model.predict_proba(vec)[0]   # [prob_genuine, prob_fake]
        fraud_prob    = float(proba[1])
        confidence    = float(max(proba))
        model_name    = type(model).__name__

    # ── Map probability → prediction label ──
    if fraud_prob >= 0.6:
        prediction = "fake"
    elif fraud_prob >= 0.35:
        prediction = "suspicious"
    else:
        prediction = "genuine"

    # ── Heuristic warnings & risk factors ──
    warnings, risk_factors = compute_heuristic_warnings(data, fraud_prob)

    # ── If model says fake but no heuristic warning – add generic one ──
    if prediction == "fake" and not warnings:
        warnings.append("ML model detected patterns commonly associated with fraudulent job postings.")

    result = {
        "prediction":       prediction,
        "fraud_probability": round(fraud_prob, 4),
        "confidence":       round(confidence, 4),
        "risk_level":       classify_risk(fraud_prob),
        "warnings":         warnings,
        "risk_factors":     risk_factors,
        "recommendation":   generate_recommendation(prediction, fraud_prob),
        "model_used":       model_name,
    }

    return jsonify(result)


# ── /metrics ─────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Returns the model comparison metrics saved by train.py.
    Falls back to empty structure if not found.
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return jsonify(json.load(f))
    return jsonify({
        "metrics":    [],
        "best_model": None,
        "message":    "No metrics found. Run: python backend/train.py"
    })


# ── Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ScamShield Flask API")
    print("  Running on http://127.0.0.1:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000,debug=True)
