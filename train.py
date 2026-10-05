# ============================================================
#  ScamShield – Fake Job Scam Detector
#  backend/train.py  |  ML Training Pipeline
#
#  Steps:
#    1. Load dataset (dataset/jobs.csv)
#    2. Clean and preprocess text using NLP
#    3. Combine important fields → TF-IDF vectors
#    4. Train Logistic Regression, Random Forest, XGBoost
#    5. Compare Accuracy, Precision, Recall, F1, ROC-AUC
#    6. Save best model + vectorizer as .pkl files
#    7. Save metrics as metrics.json for the frontend
# ============================================================

import os
import re
import json
import string
import warnings
import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection   import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Try importing XGBoost (optional dependency)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] XGBoost not installed. Skipping XGBoost model.")
    print("    Install with:  pip install xgboost")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(BASE_DIR)
DATASET_PATH  = os.path.join(PROJECT_ROOT, "dataset", "jobs.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
VECTOR_PATH   = os.path.join(BASE_DIR, "vectorizer.pkl")
METRICS_PATH  = os.path.join(BASE_DIR, "metrics.json")


# ─────────────────────────────────────────────────────────
# Step 1 – Load Dataset
# ─────────────────────────────────────────────────────────
print("\n[1/7] Loading dataset...")

if not os.path.exists(DATASET_PATH):
    print(f"\n  ✗ Dataset not found at: {DATASET_PATH}")
    print("  Please download the EMSCAD dataset from:")
    print("  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction")
    print("  Rename the file to  jobs.csv  and place it in the  dataset/  folder.\n")
    exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")
print(f"  Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────
# Step 2 – Clean Data
# ─────────────────────────────────────────────────────────
print("\n[2/7] Cleaning data...")

# Identify the label column (EMSCAD uses 'fraudulent')
if "fraudulent" in df.columns:
    label_col = "fraudulent"
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError("Could not find label column ('fraudulent' or 'label'). "
                     "Please check your CSV file.")

# Keep only rows where label is 0 or 1
df = df[df[label_col].isin([0, 1])].copy()

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed {before - len(df):,} duplicate rows.")

# Fill missing values with empty string for text columns
text_cols = ["title", "company_profile", "description", "requirements",
             "benefits", "location", "department", "salary_range",
             "employment_type", "required_experience", "required_education",
             "industry", "function"]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

print(f"  Final dataset: {len(df):,} rows")
print(f"  Class distribution:")
print(df[label_col].value_counts().to_string())
print(f"  Fraud rate: {df[label_col].mean()*100:.2f}%")


# ─────────────────────────────────────────────────────────
# Step 3 – Combine Text Fields
# ─────────────────────────────────────────────────────────
print("\n[3/7] Combining text fields...")

# Map available columns to combined text
def combine_fields(row):
    """Combine all relevant text columns into one string."""
    parts = []
    for col in ["title", "company_profile", "description", "requirements",
                "benefits", "location", "salary_range", "employment_type",
                "required_experience", "industry"]:
        if col in row.index and str(row[col]).strip():
            parts.append(str(row[col]))
    return " ".join(parts)

df["combined_text"] = df.apply(combine_fields, axis=1)
print("  Combined fields into single 'combined_text' column.")


# ─────────────────────────────────────────────────────────
# Step 4 – NLP Preprocessing
# ─────────────────────────────────────────────────────────
print("\n[4/7] NLP preprocessing...")

# Lightweight stopwords list (no external dependency)
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
    "just","s","t","now","d","ll","m","o","re","ve","y"
}

def clean_text(text: str) -> str:
    """Standard NLP cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # Remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                   # Remove HTML
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

df["clean_text"] = df["combined_text"].apply(clean_text)
print("  Cleaned text: lowercase, removed URLs, HTML, punctuation, stopwords.")

# Remove empty rows after cleaning
df = df[df["clean_text"].str.strip() != ""]
print(f"  Rows after cleaning: {len(df):,}")


# ─────────────────────────────────────────────────────────
# Step 5 – TF-IDF Vectorisation
# ─────────────────────────────────────────────────────────
print("\n[5/7] TF-IDF vectorization...")

X = df["clean_text"]
y = df[label_col].astype(int)

# Train / test split (80/20, stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF with unigrams + bigrams, max 15,000 features
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,       # Apply log normalisation
    min_df=2,                # Ignore very rare terms
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"  TF-IDF matrix: {X_train_vec.shape[0]:,} train, {X_test_vec.shape[0]:,} test, {X_train_vec.shape[1]:,} features.")

# Save vectorizer
joblib.dump(tfidf, VECTOR_PATH)
print(f"  Saved vectorizer → {VECTOR_PATH}")


# ─────────────────────────────────────────────────────────
# Step 6 – Train Models
# ─────────────────────────────────────────────────────────
print("\n[6/7] Training models...")

def evaluate(model, X_test, y_test):
    """Return a dict of evaluation metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }

results  = []
trained  = {}

# ── Model 1: Logistic Regression ──
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)
lr.fit(X_train_vec, y_train)
m = evaluate(lr, X_test_vec, y_test)
m["model"] = "Logistic Regression"
results.append(m)
trained["Logistic Regression"] = lr
print(f"    Accuracy: {m['accuracy']*100:.2f}%  |  F1: {m['f1_score']*100:.2f}%  |  ROC-AUC: {m['roc_auc']*100:.2f}%")
print(classification_report(y_test, lr.predict(X_test_vec), target_names=["Genuine", "Fake"]))

# ── Model 2: Random Forest ──
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_vec, y_train)
m = evaluate(rf, X_test_vec, y_test)
m["model"] = "Random Forest"
results.append(m)
trained["Random Forest"] = rf
print(f"    Accuracy: {m['accuracy']*100:.2f}%  |  F1: {m['f1_score']*100:.2f}%  |  ROC-AUC: {m['roc_auc']*100:.2f}%")

# ── Model 3: XGBoost (if available) ──
if XGBOOST_AVAILABLE:
    print("  Training XGBoost...")
    # Calculate scale_pos_weight to handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pw  = neg_count / pos_count if pos_count > 0 else 1

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pw,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_vec, y_train)
    m = evaluate(xgb, X_test_vec, y_test)
    m["model"] = "XGBoost"
    results.append(m)
    trained["XGBoost"] = xgb
    print(f"    Accuracy: {m['accuracy']*100:.2f}%  |  F1: {m['f1_score']*100:.2f}%  |  ROC-AUC: {m['roc_auc']*100:.2f}%")


# ─────────────────────────────────────────────────────────
# Step 7 – Select Best Model & Save
# ─────────────────────────────────────────────────────────
print("\n[7/7] Selecting & saving best model...")

# Best model = highest ROC-AUC score
best_result = max(results, key=lambda x: x["roc_auc"])
best_name   = best_result["model"]
best_model  = trained[best_name]

joblib.dump(best_model, MODEL_PATH)
print(f"  Best model: {best_name} (ROC-AUC = {best_result['roc_auc']*100:.2f}%)")
print(f"  Saved model → {MODEL_PATH}")

# Save metrics for the frontend /metrics endpoint
metrics_payload = {
    "metrics":    results,
    "best_model": best_name,
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics_payload, f, indent=2)
print(f"  Saved metrics → {METRICS_PATH}")

print("\n" + "="*55)
print("  Training complete!")
print("  Now run:  python backend/app.py")
print("="*55 + "\n")
