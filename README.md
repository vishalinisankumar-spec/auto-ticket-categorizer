# Auto Email / Ticket Categorizer

A lightweight NLP classifier that reads an incoming support ticket (subject + body) and predicts which department should handle it — **Billing**, **Technical**, **HR**, or **General** — in real time.

Built as part of an AI/ML intern technical assessment, focused on making a fast, explainable, defensible routing decision on every incoming ticket rather than chasing the highest possible accuracy on a toy dataset.

## What it does

1. **Preprocesses** raw ticket text — lowercasing, stripping URLs/emails/punctuation.
2. **Vectorizes** it with TF-IDF (unigrams + bigrams), which downweights filler words ("please", "thanks") and upweights category-distinctive words ("invoice", "password", "leave").
3. **Trains and compares** two models — Multinomial Naive Bayes and Logistic Regression — and picks whichever scores higher on held-out data.
4. **Evaluates** with accuracy, a full precision/recall report, and a confusion matrix.
5. **Predicts** the category for new, unseen tickets in real time, with:
   - a **confidence score** alongside the label
   - a **60% confidence threshold** — anything below that gets routed to a human-review queue instead of being auto-assigned, so a low-confidence guess never silently misroutes a real ticket
   - a **priority tag** (`urgent` / `normal`) from simple keyword rules ("down", "urgent", "not working", etc.)

## Why Naive Bayes / Logistic Regression over a deep model

The brief called for something lightweight and real-time. A TF-IDF + linear classifier:
- runs inference in milliseconds on CPU, no GPU needed
- is easy to retrain as new tickets/categories come in
- is interpretable — you can see which words drive each prediction, which matters when non-ML staff need to trust a routing decision

A transformer-based model might edge out accuracy by a few points but adds latency and deployment weight that isn't justified at this data scale.

## Dataset

`tickets.csv` — a small, hand-curated set of realistic support tickets (subject + body + category) across the four departments. Built manually for this assessment. With a larger set of real historical tickets, accuracy would be expected to improve — a known limitation of any small labeled dataset, not something to paper over.

## Files

- `ticket_categorizer.py` — full pipeline: preprocessing, training, evaluation, and prediction, all in one script
- `tickets.csv` — labeled training data

## Running it

```bash
pip install pandas scikit-learn
python ticket_categorizer.py
```

This prints model comparison metrics, a confusion matrix, and live predictions (with confidence + priority + human-review flag) on 5 sample tickets.

## What I'd improve with more time/data

- Swap in real historical tickets instead of a hand-curated sample
- Log low-confidence predictions and feed corrected labels back into retraining (active learning loop)
- Add basic typo/spelling tolerance, since real users don't type cleanly
- Multi-label support for tickets that genuinely span two departments
