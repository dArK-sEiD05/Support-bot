"""
Script 4: Train an intent classifier

Uses TF-IDF + Logistic Regression on labeled customer tweets.
Outputs:
  - data/intent_model.pkl        (trained classifier pipeline)
  - data/classification_report.txt

"""

import json
import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def clean_text(text):
    text = re.sub(r'@\S+', '', text)           # remove @mentions
    text = re.sub(r'https?://\S+', '', text)    # remove URLs
    text = re.sub(r'#\S+', '', text)            # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)     # keep only letters
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def main():
    # Load labeled tweets
    labeled_path = os.path.join(DATA_DIR, 'labeled_tweets.json')
    with open(labeled_path, 'r') as f:
        tweets = json.load(f)

    # Filter out 'other' — we only train on labeled intents
    labeled = [(t['text'], t['intent']) for t in tweets if t['intent'] != 'other']
    print(f"Training on {len(labeled):,} labeled tweets (excluding 'other')")

    texts = [clean_text(t[0]) for t in labeled]
    labels = [t[1] for t in labeled]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight='balanced',
            random_state=42,
        ))
    ])

    # Train
    print("\nTraining...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\n  Test Accuracy: {accuracy:.1%}")

    # Cross-validation
    print("\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"\n--- Classification Report ---\n")
    print(report)

    # Save report
    report_path = os.path.join(DATA_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})\n\n")
        f.write(report)
    print(f"  Saved report -> {report_path}")

    # Save model
    model_path = os.path.join(DATA_DIR, 'intent_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"  Saved model  -> {model_path}")

    # Quick test
    test_queries = [
        "my spotify keeps crashing every time I open it",
        "I was charged twice this month for premium",
        "how do I connect spotify to my bluetooth speaker",
        "why is this song greyed out I can't play it",
        "my downloads all disappeared overnight",
        "shuffle keeps playing the same 10 songs",
        "the sound quality is really bad on my headphones",
        "I keep seeing the same annoying ad over and over",
        "someone hacked my account and changed my password",
        "can you add a sleep timer feature",
        "is spotify available in pakistan",
        "my discover weekly is terrible this week",
    ]

    print(f"\n--- Quick Test ---\n")
    for query in test_queries:
        pred = pipeline.predict([clean_text(query)])[0]
        proba = pipeline.predict_proba([clean_text(query)])[0]
        confidence = max(proba)
        print(f"  [{pred:<25} {confidence:.0%}]  \"{query}\"")


if __name__ == '__main__':
    main()
