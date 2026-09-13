"""
Script 6: Full bot test — end-to-end RAG pipeline

Flow:
  User query → Intent classifier → KB retriever → Response generator

Tests with 20 real-world style queries and evaluates quality.

"""

import json
import os
import pickle
import re
import sys
import textwrap

# Add scripts dir to path so we can import kb_retriever
sys.path.insert(0, os.path.dirname(__file__))
from kb_retriever import KBRetriever  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
KB_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')


def clean_text(text):
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


class SpotifyBot:
    def __init__(self):
        # Load intent classifier
        model_path = os.path.join(DATA_DIR, 'intent_model.pkl')
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)

        # Load retriever
        retriever_path = os.path.join(KB_DIR, 'retriever.pkl')
        with open(retriever_path, 'rb') as f:
            self.retriever = pickle.load(f)

        # Load diagnostics
        diag_path = os.path.join(KB_DIR, 'diagnostics.json')
        with open(diag_path, 'r') as f:
            self.diagnostics = json.load(f)

    def classify_intent(self, query):
        cleaned = clean_text(query)
        intent = self.classifier.predict([cleaned])[0]
        proba = self.classifier.predict_proba([cleaned])[0]
        confidence = max(proba)
        return intent, confidence

    def retrieve_knowledge(self, query, intent, top_k=2):
        return self.retriever.retrieve(query, intent=intent, top_k=top_k)

    def get_diagnostic_questions(self, intent):
        per_intent = self.diagnostics.get('per_intent', {})
        return per_intent.get(intent, self.diagnostics.get('standard_flow', []))

    def generate_response(self, query, intent, confidence, kb_results):
        """Generate a response from intent + retrieved KB articles."""

        parts = []

        # Opening based on intent
        openings = {
            'playback_issue': "Sorry to hear you're having playback issues!",
            'account_access': "We understand how frustrating it is not being able to access your account.",
            'subscription_billing': "Let me help you with your billing concern.",
            'content_availability': "We understand it's frustrating when content isn't available.",
            'download_offline': "Let me help with your download/offline issue.",
            'shuffle_queue': "Let me help you with shuffle and queue settings.",
            'device_compatibility': "Let's get your device working with Spotify.",
            'feature_request': "Thanks for the suggestion!",
            'audio_quality': "Let's get your audio quality sorted out.",
            'playlist_library': "Let me help with your playlist/library issue.",
            'ads_complaints': "We understand ads can be frustrating.",
            'country_region': "Let me check on availability for you.",
        }
        parts.append(openings.get(intent, "Thanks for reaching out!"))

        # Extract actionable content from top KB result
        if kb_results:
            top = kb_results[0]
            text = top['text']

            # Pull out steps if they exist
            steps = []
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('- ') and len(line) > 5:
                    step = line[2:].strip()
                    # Skip meta lines
                    if any(skip in step.lower() for skip in ['intent:', 'topic:']):
                        continue
                    steps.append(step)

            if steps:
                parts.append("\nHere's what you can try:")
                for i, step in enumerate(steps[:6], 1):
                    parts.append(f"  {i}. {step}")

            # Add URL if available
            if top.get('url'):
                parts.append(f"\nMore info: {top['url']}")

        # Add diagnostic questions if confidence is low or issue is vague
        if confidence < 0.6:
            diag = self.get_diagnostic_questions(intent)
            if diag:
                parts.append("\nTo help further, could you tell me:")
                for q in diag[:3]:
                    parts.append(f"  • {q}")

        return '\n'.join(parts)

    def respond(self, query):
        """Full pipeline: classify → retrieve → respond."""
        intent, confidence = self.classify_intent(query)
        kb_results = self.retrieve_knowledge(query, intent)
        response = self.generate_response(query, intent, confidence, kb_results)
        return {
            'query': query,
            'intent': intent,
            'confidence': round(confidence, 3),
            'kb_top_match': kb_results[0]['topic'] if kb_results else None,
            'kb_score': kb_results[0]['score'] if kb_results else 0,
            'response': response,
        }


# ── Test Queries ──
# Mix of real tweet styles + clean questions covering all intents

TEST_QUERIES = [
    # playback_issue
    {"q": "spotify keeps crashing every time i open it on my iphone", "expected": "playback_issue"},
    {"q": "music just stops playing randomly after a few songs", "expected": "playback_issue"},

    # account_access
    {"q": "someone hacked my account and is playing music on it", "expected": "account_access"},
    {"q": "i forgot my password and cant log in anymore", "expected": "account_access"},

    # subscription_billing
    {"q": "i was charged $12.99 but i cancelled last month", "expected": "subscription_billing"},
    {"q": "how do i switch from family plan to individual", "expected": "subscription_billing"},

    # device_compatibility
    {"q": "spotify wont connect to my bluetooth speaker", "expected": "device_compatibility"},
    {"q": "how do i play spotify on my ps4 while gaming", "expected": "device_compatibility"},

    # download_offline
    {"q": "all my downloaded songs disappeared overnight", "expected": "download_offline"},
    {"q": "how many songs can i download for offline listening", "expected": "download_offline"},

    # shuffle_queue
    {"q": "shuffle keeps playing the same 5 songs over and over", "expected": "shuffle_queue"},
    {"q": "how do i clear my play queue on mobile", "expected": "shuffle_queue"},

    # audio_quality
    {"q": "the sound quality is really bad even on high setting", "expected": "audio_quality"},
    {"q": "how do i enable lossless audio on spotify", "expected": "audio_quality"},

    # ads_complaints
    {"q": "im getting an ad after every single song this is insane", "expected": "ads_complaints"},
    {"q": "i watched the video ad but didnt get 30 min ad free", "expected": "ads_complaints"},

    # content_availability
    {"q": "why is taylor swifts old album greyed out i cant play it", "expected": "content_availability"},
    {"q": "this song was on spotify last week now its gone", "expected": "content_availability"},

    # country_region
    {"q": "when will spotify be available in pakistan", "expected": "country_region"},

    # playlist_library
    {"q": "my discover weekly playlist is giving me terrible recommendations", "expected": "playlist_library"},
]


def main():
    print("Loading bot components...\n")
    bot = SpotifyBot()

    correct = 0
    total = len(TEST_QUERIES)
    results = []

    print("=" * 70)
    print("  SPOTIFY SUPPORT BOT — END-TO-END TEST")
    print("=" * 70)

    for i, test in enumerate(TEST_QUERIES, 1):
        result = bot.respond(test['q'])
        expected = test['expected']
        match = result['intent'] == expected

        if match:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"\n{'─' * 70}")
        print(f"  Test {i:>2}/{total}  {status}  Intent: {result['intent']:<25} "
              f"(expected: {expected})")
        print(f"  Confidence: {result['confidence']:.0%}  |  "
              f"KB match: {result['kb_top_match']} (score: {result['kb_score']:.3f})")
        print(f"{'─' * 70}")
        print(f"  USER: {test['q']}")
        print()
        # Indent the response
        for line in result['response'].split('\n'):
            print(f"  BOT:  {line}")

        results.append({
            'test_num': i,
            'query': test['q'],
            'expected_intent': expected,
            'predicted_intent': result['intent'],
            'correct': match,
            'confidence': result['confidence'],
            'kb_topic': result['kb_top_match'],
            'kb_score': result['kb_score'],
        })

    # Summary
    accuracy = correct / total
    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {correct}/{total} correct ({accuracy:.0%} accuracy)")
    print(f"{'=' * 70}")

    # Per-intent breakdown
    from collections import defaultdict
    intent_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        intent_results[r['expected_intent']]['total'] += 1
        if r['correct']:
            intent_results[r['expected_intent']]['correct'] += 1

    print(f"\n  {'Intent':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'-' * 50}")
    for intent in sorted(intent_results):
        d = intent_results[intent]
        acc = d['correct'] / d['total'] if d['total'] > 0 else 0
        print(f"  {intent:<25} {d['correct']:>8} {d['total']:>6} {acc:>8.0%}")

    # Show failures
    failures = [r for r in results if not r['correct']]
    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f"    \"{f['query']}\"")
            print(f"      expected: {f['expected_intent']}  |  got: {f['predicted_intent']} ({f['confidence']:.0%})")

    # Confidence stats
    confidences = [r['confidence'] for r in results]
    avg_conf = sum(confidences) / len(confidences)
    min_conf = min(confidences)
    print(f"\n  Avg confidence: {avg_conf:.0%}  |  Min: {min_conf:.0%}")

    # KB retrieval quality
    kb_scores = [r['kb_score'] for r in results]
    avg_kb = sum(kb_scores) / len(kb_scores)
    print(f"  Avg KB score:   {avg_kb:.3f}")

    # Save results
    results_path = os.path.join(DATA_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved test results -> {results_path}")


if __name__ == '__main__':
    main()
