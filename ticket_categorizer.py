"""
Auto Email / Ticket Categorizer — AI/ML Intern Assessment (Fobes Skill iTech)

HOW TO USE:
1. Set DATA_PATH below to your provided CSV.
2. Check COLUMN NAMES in the "CONFIG" section match your actual file
   (they gave you subject + body + category — adjust the 3 names if different).
3. Run top to bottom. Everything — preprocessing, training, evaluation,
   live prediction, and all 5 bonus objectives — is in this one file.

Covers:
  Core: text preprocessing -> TF-IDF -> Naive Bayes (justified) -> accuracy/
        precision/recall/confusion matrix -> predict 5 new sample tickets
  Bonus 1: confidence score output
  Bonus 2: "needs human review" threshold (<60%)
  Bonus 3: priority tagging (urgent/normal via keyword rules)
  Bonus 4: mini live CLI demo (type a ticket, get instant routing)
  Bonus 5: reflection note (bottom of file)
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ============================================================
# CONFIG — adjust these 4 lines to match your actual dataset
# ============================================================
DATA_PATH = "tickets.csv"      # path to the CSV they gave you
SUBJECT_COL = "subject"        # column name for the subject line
BODY_COL = "body"              # column name for the ticket body
LABEL_COL = "category"         # column name for the true category
HUMAN_REVIEW_THRESHOLD = 0.60  # bonus 2: below this confidence -> manual queue


# ============================================================
# STEP 1 — Load data
# ============================================================
df = pd.read_csv(DATA_PATH)

# Combine subject + body into one text field. Subject lines are short but
# often carry a strong signal ("Refund request", "Login issue") so we don't
# want to throw that away — just concatenate before vectorizing.
df["text"] = df[SUBJECT_COL].fillna("") + " " + df[BODY_COL].fillna("")


# ============================================================
# STEP 2 — Text preprocessing
# ============================================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # strip URLs
    text = re.sub(r"\S+@\S+", " ", text)                # strip emails
    text = re.sub(r"[^a-z\s]", " ", text)                # strip punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text

df["clean_text"] = df["text"].apply(clean_text)


# ============================================================
# STEP 3 — Feature representation: TF-IDF
# ============================================================
# TF-IDF over raw counts because it downweights words that appear in almost
# every ticket (e.g. "please", "thanks", "help") and upweights words that
# are actually distinctive for a category (e.g. "invoice", "password",
# "leave", "salary") — which is exactly the signal a router needs.
vectorizer = TfidfVectorizer(
    stop_words="english",   # drop common English filler words
    ngram_range=(1, 2),     # unigrams + bigrams (catches phrases like "not working")
    min_df=1,
)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_text"], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ============================================================
# STEP 4 — Model choice + training
# ============================================================
# Why Multinomial Naive Bayes as the primary model:
# - It's built for exactly this input shape: sparse word-count/TF-IDF
#   features over short documents. The "naive" conditional-independence
#   assumption between words is wrong in a strict sense, but for bag-of-words
#   text classification it works well in practice and needs very little data
#   to start generalizing — a good fit for a "small labeled dataset."
# - It's fast to train/predict, which matters for the "sits in front of a
#   live ticket queue" real-time requirement.
# We also train Logistic Regression as a comparison point, since it
# typically handles correlated/overlapping words a bit better than NB's
# independence assumption, and pick whichever scores higher on this data.
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, acc)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, preds))
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(df[LABEL_COL].unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))

best_name = max(results, key=lambda k: results[k][1])
model = results[best_name][0]
print(f"\n>>> Using {best_name} for predictions (higher accuracy on this data)")


# ============================================================
# STEP 5 — Priority tagging (Bonus 3): simple keyword rule layer
# ============================================================
URGENT_KEYWORDS = {
    "urgent", "asap", "immediately", "down", "not working", "broken",
    "critical", "emergency", "outage", "cannot access", "can't access",
    "blocked", "failing", "failed", "crash", "crashed",
}

def tag_priority(raw_text: str) -> str:
    text_lower = raw_text.lower()
    return "urgent" if any(kw in text_lower for kw in URGENT_KEYWORDS) else "normal"


# ============================================================
# STEP 6 — Predict function with confidence + human-review fallback
# (Bonus 1: confidence score, Bonus 2: human review threshold)
# ============================================================
def predict_ticket(raw_text: str) -> dict:
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_category, top_conf = ranked[0]
    else:
        top_category = model.predict(vec)[0]
        top_conf = None  # some models don't expose probabilities

    needs_human_review = (top_conf is None) or (top_conf < HUMAN_REVIEW_THRESHOLD)

    return {
        "text": raw_text,
        "predicted_category": top_category if not needs_human_review else "UNCERTAIN — human review",
        "confidence": round(float(top_conf), 3) if top_conf is not None else None,
        "priority": tag_priority(raw_text),
        "needs_human_review": needs_human_review,
    }


# ============================================================
# STEP 7 — Test on 5 new, hand-written sample tickets (core requirement)
# ============================================================
sample_tickets = [
    "My salary for this month hasn't been credited yet, please check.",
    "The app crashes every time I try to upload a file, this is urgent.",
    "I was charged twice for my subscription this month, need a refund.",
    "Can you tell me your office working hours?",
    "I want to apply for maternity leave, what's the process?",
]

print("\n" + "=" * 60)
print("LIVE PREDICTIONS ON 5 NEW SAMPLE TICKETS")
print("=" * 60)
for ticket in sample_tickets:
    result = predict_ticket(ticket)
    print(f"\nTicket: {result['text']}")
    print(f"  -> Category: {result['predicted_category']}")
    print(f"  -> Confidence: {result['confidence']}")
    print(f"  -> Priority: {result['priority']}")
    print(f"  -> Needs human review: {result['needs_human_review']}")


# ============================================================
# STEP 8 (Bonus 4) — Mini live CLI demo
# Uncomment to run interactively: type a ticket, get instant routing.
# ============================================================
if __name__ == "__main__":
    print("\nType a support ticket (or 'quit' to exit):")
    while True:
         user_input = input("\n> ")
        if user_input.strip().lower() == "quit":
             break
        result = predict_ticket(user_input)
        print(f"Category: {result['predicted_category']} | "
              f"Confidence: {result['confidence']} | "
              f"Priority: {result['priority']}")


# ============================================================
# REFLECTION NOTE (Bonus 5)
# ============================================================
"""
With more data, I'd want real historical tickets rather than relying only
on the small provided set — a handful of examples per category makes it easy
for the model to latch onto template phrasing rather than genuine topic
signal, so accuracy on the held-out split can overstate real-world
performance. I'd also add a confusion-matrix-driven error review: the
categories the model confuses most (likely General vs. Technical, since
vague tickets can read as either) tell you where more labeled examples or
clearer category definitions are needed. With more time, I'd add basic
misspelling handling, since real users don't type cleanly, and I'd log
low-confidence predictions in production so corrected labels can be fed
back into retraining over time.
"""
