"""
Spotify Support Bot — LLM-powered (Groq)

Flow:
  User query → Urgency detection → Intent classification → Escalation check
  → KB retrieval + Historical retrieval → LLM response generation

Run: python3 bot.py
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))


def strip_think(text):
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

# Add scripts dir for KBRetriever
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
from kb_retriever import KBRetriever  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
KB_DIR = os.path.join(BASE_DIR, 'knowledge_base')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

INTENTS = [
    "playback_issue",
    "account_access",
    "subscription_billing",
    "content_availability",
    "download_offline",
    "shuffle_queue",
    "device_compatibility",
    "feature_request",
    "audio_quality",
    "playlist_library",
    "ads_complaints",
    "country_region",
    "other",
]

INTENT_DESCRIPTIONS = """
- playback_issue: App crashes, freezing, music stops, buffering, not playing, no sound, glitches
- account_access: Can't log in, forgot password, hacked account, locked out, unauthorized access
- subscription_billing: Charges, refunds, cancel premium, change plan, payment failed, pricing, free trial
- content_availability: Songs/albums missing, greyed out, removed, not available, artist content gone
- download_offline: Downloads disappeared, can't download, offline mode, storage, sync issues
- shuffle_queue: Shuffle not random, same songs repeating, can't turn off shuffle, queue management, autoplay
- device_compatibility: Bluetooth, speakers, car, PS4/PS5, Xbox, Alexa, Chromecast, Roku, smart TV, Spotify Connect, Apple Watch
- feature_request: Suggesting new features, requesting improvements, wish list
- audio_quality: Sound quality bad, bitrate, lossless, equalizer, distorted, volume issues
- playlist_library: Discover Weekly, Daily Mix, playlist issues, library, saved songs, recommendations
- ads_complaints: Too many ads, ad-free not working, annoying ads, 30-min ad-free didn't work
- country_region: Spotify not available in country, region restrictions, VPN, when launching
- other: Greetings, off-topic, unclear intent
"""

# ── Urgency Detection ──

CRITICAL_PATTERNS = [
    'wtf', 'fuck', 'fucking', 'bullshit', 'scam', 'fraud', 'steal', 'stole',
    'stolen', 'sue', 'lawyer', 'legal', 'report you', 'calling my bank',
    'switching to apple music', 'moving to apple music', 'cancelling forever',
    'worst service ever', 'absolute garbage', 'i hate you',
]

HIGH_PATTERNS = [
    'ridiculous', 'unacceptable', 'worst', 'terrible', 'horrible', 'useless',
    'pathetic', 'disgusting', 'outrageous', 'furious', 'livid', 'done with',
    'fed up', 'sick of', 'had enough', 'never again',
]

MEDIUM_PATTERNS = [
    'annoying', 'frustrating', 'disappointed', 'come on', 'seriously',
    'not happy', 'unhappy', 'really bad', 'so tired of', 'ugh',
]

# Intents that always escalate immediately
INSTANT_ESCALATE_INTENTS = {
    'account_access': ['hacked', 'stolen', 'unauthorized', 'someone else', 'fraud',
                       'changed my email', 'changed my password', 'compromised'],
}

# Intents where the bot should suggest human support (but still try to help)
SUGGEST_HUMAN_INTENTS = ['subscription_billing']

FAILURE_SIGNALS = [
    "still not working", "didn't work", "doesn't work", "didnt work",
    "still broken", "same issue", "same problem", "not fixed",
    "still happening", "nope", "no luck", "still crashing",
    "that didn't help", "nothing works", "still the same",
    "tried that", "already tried", "already did that",
]

MAX_TURNS = 5
MAX_FAILED_FIXES = 3


def detect_urgency(text):
    """Detect customer urgency/frustration level."""
    text_lower = text.lower()
    is_all_caps = len(text) > 10 and text == text.upper()

    if is_all_caps or any(p in text_lower for p in CRITICAL_PATTERNS):
        return 'CRITICAL'
    if any(p in text_lower for p in HIGH_PATTERNS):
        return 'HIGH'
    if any(p in text_lower for p in MEDIUM_PATTERNS):
        return 'MEDIUM'
    return 'LOW'


def urgency_emoji(level):
    return {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'CRITICAL': '🔴'}[level]


def should_instant_escalate(intent, text):
    """Check if this query should be escalated immediately."""
    text_lower = text.lower()
    if intent in INSTANT_ESCALATE_INTENTS:
        triggers = INSTANT_ESCALATE_INTENTS[intent]
        if any(t in text_lower for t in triggers):
            return True, "Security/fraud issue — requires human agent with account access"
    return False, ""


def detect_failure(text):
    """Check if the user is saying a suggested fix didn't work."""
    text_lower = text.lower()
    return any(f in text_lower for f in FAILURE_SIGNALS)


class SpotifyBot:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = "qwen/qwen3.8-27b"

        # Load KB retriever (embeddings + ChromaDB)
        self.retriever = KBRetriever(load_existing=True)

        # Load historical response collection from ChromaDB
        import chromadb
        from sentence_transformers import SentenceTransformer
        chroma_dir = os.path.join(KB_DIR, 'chroma_db')
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.historical_collection = chroma_client.get_collection("spotify_historical")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load diagnostics
        diag_path = os.path.join(KB_DIR, 'diagnostics.json')
        with open(diag_path, 'r') as f:
            self.diagnostics = json.load(f)

        # Conversation state
        self.history = []
        self.turn_count = 0
        self.failed_fixes = 0
        self.attempted_fixes = []
        self.current_intent = None
        self.escalated = False

    def classify_intent(self, query):
        prompt = f"""You are an intent classifier for Spotify customer support.

Given a user message, classify it into exactly ONE of these intents:
{INTENT_DESCRIPTIONS}

User message: "{query}"

Respond with ONLY the intent label (e.g., "playback_issue"). Nothing else. /no_think"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30,
        )
        raw = strip_think(response.choices[0].message.content).strip().lower()
        for i in INTENTS:
            if i in raw:
                return i
        return "other"

    def retrieve_knowledge(self, query, intent, top_k=3):
        return self.retriever.retrieve(query, intent=intent, top_k=top_k)

    def retrieve_historical(self, query, intent, top_k=3):
        """Retrieve similar resolved conversations from historical data."""
        query_embedding = self.embed_model.encode([query]).tolist()
        try:
            results = self.historical_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"intent": intent},
            )
        except Exception:
            results = self.historical_collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )

        output = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                dist = results['distances'][0][i] if results['distances'] else 0
                output.append({
                    'score': round(1 - dist, 4),
                    'text': doc,
                })
        return output

    def get_diagnostics(self, intent):
        per_intent = self.diagnostics.get('per_intent', {})
        return per_intent.get(intent, self.diagnostics.get('standard_flow', []))

    def build_escalation_summary(self, query, intent, urgency):
        """Build a handoff summary for the human agent."""
        # Collect user info from conversation
        user_messages = [m['content'] for m in self.history if m['role'] == 'user']

        summary_parts = [
            f"Issue type: {intent}",
            f"Urgency: {urgency}",
            f"Turns in conversation: {self.turn_count}",
        ]

        if self.attempted_fixes:
            summary_parts.append(f"Fixes attempted: {', '.join(self.attempted_fixes)}")
            summary_parts.append(f"Fixes that failed: {self.failed_fixes}")

        summary_parts.append(f"Customer messages: {' | '.join(user_messages[-3:] + [query])}")

        return '\n     '.join(summary_parts)

    def generate_escalation_response(self, query, intent, urgency):
        """Generate an escalation message with handoff summary."""
        summary = self.build_escalation_summary(query, intent, urgency)

        prompt = f"""You are SpotBot, a Spotify support bot. You need to escalate this conversation to a human agent.

The customer's issue could not be resolved by the bot. Generate a warm, empathetic escalation message that:
1. Acknowledges their frustration (if frustrated)
2. Explains you're connecting them with a specialist
3. Includes the handoff summary below so the human agent has context
4. Provides the support link: https://support.spotify.com/contact-spotify-support/

Handoff summary:
{summary}

Be concise. /no_think"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return strip_think(response.choices[0].message.content)

    def generate_response(self, query, intent, kb_results, historical_results, urgency):
        # Build KB context
        kb_context = ""
        for i, result in enumerate(kb_results[:2], 1):
            kb_context += f"\n--- Article {i}: {result['topic']} ---\n{result['text']}\n"

        # Build historical context
        hist_context = ""
        for i, result in enumerate(historical_results[:3], 1):
            hist_context += f"\n--- Historical Conversation {i} (resolved) ---\n{result['text']}\n"

        # Get diagnostic questions
        diag = self.get_diagnostics(intent)
        diag_text = "\n".join(f"  - {q}" for q in diag) if diag else "None"

        # Build conversation history
        history_text = ""
        if self.history:
            recent = self.history[-6:]
            for msg in recent:
                role = "User" if msg['role'] == 'user' else 'Bot'
                history_text += f"{role}: {msg['content']}\n"

        # Build escalation awareness context
        escalation_context = f"""
Conversation status:
  - Turn: {self.turn_count + 1} of {MAX_TURNS} (auto-escalate at {MAX_TURNS})
  - Failed fix attempts: {self.failed_fixes} of {MAX_FAILED_FIXES} (escalate at {MAX_FAILED_FIXES})
  - Fixes already tried: {', '.join(self.attempted_fixes) if self.attempted_fixes else 'None yet'}
  - Customer urgency: {urgency}

ESCALATION RULES:
  - Do NOT suggest a fix that's already in the "Fixes already tried" list
  - If failed fixes >= {MAX_FAILED_FIXES}, do NOT suggest another fix — escalate instead
  - If this is a billing/refund/account-specific issue that needs private access, suggest human support
  - If urgency is CRITICAL, be extra empathetic"""

        suggest_human = intent in SUGGEST_HUMAN_INTENTS
        billing_note = ""
        if suggest_human:
            billing_note = "\nNOTE: This is a billing/subscription issue. After giving any general info you can, always mention that for account-specific actions they should contact support directly."

        system_prompt = f"""You are a friendly, helpful Spotify support bot. Your name is SpotBot.

RULES:
1. Be concise — max 4-5 sentences unless the user asks for detail
2. Give ACTIONABLE steps when possible (numbered steps)
3. If you can fully resolve the issue from the knowledge base, do it
4. If you need more info, ask a specific diagnostic question (pick ONE, not many)
5. Use a warm, casual tone (like SpotifyCares on Twitter)
6. Never make up information — only use what's in the knowledge base and historical responses
7. If the issue needs private account access, say: "For account-specific help, reach out to Spotify support at https://support.spotify.com/contact-spotify-support/"
8. Don't repeat yourself if the user already provided info in the conversation
9. Do NOT include any thinking or reasoning in your response — just the answer. /no_think
10. Learn from the historical conversations below — match their tone and troubleshooting patterns.
{billing_note}

Detected intent: {intent}
{escalation_context}

Knowledge base articles (use for factual accuracy):
{kb_context}

Historical resolved conversations (use for tone, troubleshooting flow, and resolution patterns):
{hist_context}

Diagnostic questions available for this intent:
{diag_text}

Previous conversation:
{history_text if history_text else "None (new conversation)"}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )

        bot_response = strip_think(response.choices[0].message.content)

        # Track attempted fixes from the bot's response
        fix_keywords = ['try ', 'restart', 'reinstall', 'log out', 'log in',
                        'uninstall', 'clear cache', 'update ', 'reset']
        for kw in fix_keywords:
            if kw in bot_response.lower() and kw.strip() not in self.attempted_fixes:
                self.attempted_fixes.append(kw.strip())

        return bot_response

    def respond(self, query):
        """Full pipeline: urgency → classify → escalation check → retrieve → respond."""

        # Step 0: Detect urgency
        urgency = detect_urgency(query)

        # Step 1: Classify intent (only on first turn, keep same intent for the conversation)
        if self.turn_count == 0:
            self.current_intent = self.classify_intent(query)
        intent = self.current_intent

        # Step 2: Check for failure signals from user
        if self.turn_count > 0 and detect_failure(query):
            self.failed_fixes += 1

        self.turn_count += 1

        # Step 3: Escalation checks
        escalate = False
        escalation_reason = ""

        # 3a: Instant escalation (security/fraud)
        should_escalate, reason = should_instant_escalate(intent, query)
        if should_escalate:
            escalate = True
            escalation_reason = reason

        # 3b: Critical urgency on first message
        if urgency == 'CRITICAL' and self.turn_count == 1:
            escalate = True
            escalation_reason = "Critical urgency detected — customer extremely frustrated"

        # 3c: Too many failed fixes
        if self.failed_fixes >= MAX_FAILED_FIXES:
            escalate = True
            escalation_reason = f"{self.failed_fixes} fix attempts failed — bot cannot resolve"

        # 3d: Max turns reached
        if self.turn_count >= MAX_TURNS and not self.escalated:
            escalate = True
            escalation_reason = f"Reached {MAX_TURNS} turns without resolution"

        # 3e: Critical urgency after failed fixes
        if urgency == 'CRITICAL' and self.failed_fixes >= 1:
            escalate = True
            escalation_reason = "Critical urgency + failed fix — escalating immediately"

        # Build response
        if escalate and not self.escalated:
            self.escalated = True
            response = self.generate_escalation_response(query, intent, urgency)
            action = "ESCALATE"
        else:
            # Normal flow: retrieve + respond
            kb_results = self.retrieve_knowledge(query, intent)
            historical_results = self.retrieve_historical(query, intent)
            response = self.generate_response(query, intent, kb_results,
                                              historical_results, urgency)
            action = "AUTO"

        # Track conversation
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})

        return {
            'intent': intent,
            'urgency': urgency,
            'action': action,
            'escalation_reason': escalation_reason,
            'turn': self.turn_count,
            'failed_fixes': self.failed_fixes,
            'response': response,
        }

    def reset(self):
        self.history = []
        self.turn_count = 0
        self.failed_fixes = 0
        self.attempted_fixes = []
        self.current_intent = None
        self.escalated = False


def main():
    print("\n" + "=" * 60)
    print("  🎵  SpotBot — Spotify Support Bot")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Commands: 'quit' to exit, 'reset' to start fresh")
    print("=" * 60 + "\n")

    bot = SpotifyBot()

    while True:
        try:
            query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Bye! 🎶")
            break

        if not query:
            continue
        if query.lower() in ('quit', 'exit', 'q'):
            print("\n  Bye! 🎶")
            break
        if query.lower() == 'reset':
            bot.reset()
            print("  [conversation reset]\n")
            continue

        result = bot.respond(query)

        # Display status line
        urg = urgency_emoji(result['urgency'])
        turn_info = f"Turn {result['turn']}/{MAX_TURNS}"
        fix_info = f"Failed fixes: {result['failed_fixes']}/{MAX_FAILED_FIXES}" if result['failed_fixes'] > 0 else ""
        action_tag = f"-> {result['action']}"
        if result['escalation_reason']:
            action_tag += f" ({result['escalation_reason']})"

        status_parts = [
            f"[{result['intent']}]",
            f"[{urg} {result['urgency']}]",
            f"[{turn_info}]",
        ]
        if fix_info:
            status_parts.append(f"[{fix_info}]")
        status_parts.append(action_tag)

        print(f"  {' '.join(status_parts)}")

        # Print response
        for line in result['response'].split('\n'):
            print(f"  Bot: {line}")
        print()


if __name__ == '__main__':
    main()
