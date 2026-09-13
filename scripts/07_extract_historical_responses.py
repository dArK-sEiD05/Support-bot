"""
Script 7: Extract resolution-confirmed historical responses

Combines Option C + D:
  - Find conversations where the customer confirmed resolution
    ("thanks", "worked", "fixed", "sorted", etc.)
  - Extract full multi-turn diagnostic flows from those conversations
  - Group by intent

Outputs:
  - knowledge_base/historical_responses.json

"""

import csv
import json
import os
import re
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
KB_DIR = os.path.join(BASE_DIR, 'knowledge_base')
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(BASE_DIR, '..', 'twcs.csv')

# Resolution confirmation signals from customer
CONFIRM_POSITIVE = [
    'thanks', 'thank you', 'thx', 'ty', 'cheers',
    'worked', 'that worked', 'it worked', 'working now',
    'fixed', 'that fixed', 'its fixed',
    'sorted', 'all sorted', 'got it sorted',
    'perfect', 'awesome', 'great', 'amazing',
    'got it', 'solved', 'resolved',
    'appreciate', 'helpful', 'that helped',
    'you rock', 'legend', 'lifesaver',
]

# Signals the issue was NOT resolved
NEGATIVE_SIGNALS = [
    'still not working', "didn't work", "doesn't work", 'didnt work',
    'still broken', 'same issue', 'same problem', 'not fixed',
    'still happening', 'nope', 'no luck', 'useless',
    'terrible', 'worst', 'give up',
]

# Agent deflection patterns to skip
DEFLECTION_PATTERNS = [
    'send us a dm', 'send a dm', 'via dm', 'please dm',
    'direct message', 'private message',
    'please follow', 'give us a follow',
    "we'll pass", 'feedback has been', 'been noted',
    'we apologize', "we're sorry",
]

# Agent resolution patterns to keep
RESOLUTION_PATTERNS = [
    'try ', 'go to ', 'tap ', 'click ', 'select ', 'open ',
    'log out', 'log in', 'sign out', 'restart', 'reinstall',
    'uninstall', 'update ', 'clear cache', 'settings >',
    'enable', 'disable', 'toggle', 'here\'s how',
    'you can ', 'you\'ll need', 'make sure', 'check if',
    'step ', 'first,', 'navigate to',
]

INTENT_KEYWORDS = {
    'playback_issue': ['not playing', "won't play", 'stops playing', 'keeps pausing',
                       "can't play", 'crashes', 'freezes', 'not working', 'app crashes',
                       'stopped working', 'music stops', 'black screen', 'glitch'],
    'account_access': ["can't log in", "can't login", "can't sign in", 'locked out',
                       'password', 'hacked', "can't access", 'unauthorized', 'stolen',
                       'compromised', 'logged out'],
    'subscription_billing': ['charged', 'billing', 'payment', 'refund', 'subscription',
                             'cancel', 'premium', 'free trial', 'student discount',
                             'family plan', 'double charged', 'money', 'price', 'plan'],
    'content_availability': ['not available', 'missing', 'removed', "can't find",
                             'greyed out', 'grayed out', 'no longer available', 'taken down'],
    'download_offline': ['download', 'offline', 'downloaded songs', 'downloads disappeared',
                         "can't download", 'offline mode', 'storage'],
    'shuffle_queue': ['shuffle', 'queue', 'repeat', 'play in order', 'random',
                      'same songs', 'autoplay'],
    'device_compatibility': ['bluetooth', 'chromecast', 'alexa', 'echo', 'apple watch',
                             'car', 'roku', 'ps4', 'ps5', 'playstation', 'xbox',
                             'smart tv', 'google home', 'fire stick', 'sonos', 'connect',
                             'speaker', 'airplay'],
    'audio_quality': ['sound quality', 'audio quality', 'low quality', 'bitrate',
                      'distorted', 'lossless', 'equalizer', 'volume'],
    'playlist_library': ['playlist', 'library', 'discover weekly', 'daily mix',
                         'release radar', 'recommended', 'saved songs', 'liked songs'],
    'ads_complaints': ['ads', 'ad ', 'advertisement', 'too many ads', 'ad free',
                       'commercial', '30 minutes'],
    'country_region': ['country', 'region', 'not available in my', 'launch in',
                       'available in'],
    'feature_request': ['wish', 'please add', 'can you add', 'suggestion', 'idea',
                        'why can\'t'],
}


def classify_text(text):
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[intent] = hits
    if not scores:
        return None
    return max(scores, key=scores.get)


def is_english(text):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.8


def is_resolution_response(text):
    text_lower = text.lower()
    if any(d in text_lower for d in DEFLECTION_PATTERNS):
        return False
    return any(r in text_lower for r in RESOLUTION_PATTERNS)


def is_confirmed_resolved(text):
    text_lower = text.lower()
    if any(n in text_lower for n in NEGATIVE_SIGNALS):
        return False
    return any(c in text_lower for c in CONFIRM_POSITIVE)


def load_tweets(csv_path):
    tweets = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets[row['tweet_id'].strip()] = {
                'tweet_id': row['tweet_id'].strip(),
                'author_id': row['author_id'].strip(),
                'inbound': row['inbound'].strip() == 'True',
                'created_at': row['created_at'].strip(),
                'text': row['text'].strip(),
                'response_tweet_id': row['response_tweet_id'].strip(),
                'in_response_to_tweet_id': row['in_response_to_tweet_id'].strip(),
            }
    return tweets


def follow_chain(start_id, tweets, visited):
    chain = []
    queue = [start_id]
    while queue:
        tid = queue.pop(0)
        if tid in visited or tid not in tweets:
            continue
        visited.add(tid)
        t = tweets[tid]
        chain.append(t)
        if t['response_tweet_id']:
            for rid in t['response_tweet_id'].split(','):
                rid = rid.strip()
                if rid and rid not in visited and rid in tweets:
                    queue.append(rid)
        if t['in_response_to_tweet_id']:
            irt = t['in_response_to_tweet_id'].strip()
            if irt and irt not in visited and irt in tweets:
                queue.append(irt)
    return chain


def main():
    print("Loading tweets...")
    tweets = load_tweets(CSV_PATH)
    print(f"  Loaded {len(tweets):,} tweets")

    # Find SpotifyCares conversation starters
    starters = []
    for tid, t in tweets.items():
        if t['inbound'] and not t['in_response_to_tweet_id']:
            resp_ids = [r.strip() for r in t['response_tweet_id'].split(',') if r.strip()]
            for rid in resp_ids:
                if rid in tweets and tweets[rid]['author_id'] == 'SpotifyCares':
                    starters.append(tid)
                    break

    print(f"  Found {len(starters):,} conversation starters")

    # Process each conversation
    all_visited = set()
    resolved_conversations = defaultdict(list)
    total_convs = 0
    resolved_count = 0
    has_resolution_steps = 0

    for tid in starters:
        if tid in all_visited:
            continue

        chain = follow_chain(tid, tweets, set())
        chain_ids = set(c['tweet_id'] for c in chain)
        if chain_ids & all_visited:
            continue
        all_visited.update(chain_ids)

        # Skip non-English
        if not all(is_english(c['text']) for c in chain):
            continue

        # Need at least 3 messages for a meaningful flow
        if len(chain) < 3:
            continue

        total_convs += 1
        chain.sort(key=lambda x: x['created_at'])

        # Classify intent from first customer message
        first_customer = next((c for c in chain if c['inbound']), None)
        if not first_customer:
            continue
        intent = classify_text(first_customer['text'])
        if not intent:
            continue

        # Check if any agent reply has actual resolution steps
        agent_msgs = [c for c in chain if c['author_id'] == 'SpotifyCares']
        has_steps = any(is_resolution_response(a['text']) for a in agent_msgs)
        if not has_steps:
            continue
        has_resolution_steps += 1

        # Check if customer confirmed resolution
        customer_msgs = [c for c in chain if c['inbound']]
        # Look at messages AFTER the first agent resolution
        first_resolution_idx = None
        for i, msg in enumerate(chain):
            if msg['author_id'] == 'SpotifyCares' and is_resolution_response(msg['text']):
                first_resolution_idx = i
                break

        if first_resolution_idx is None:
            continue

        # Check customer messages after the resolution
        confirmed = False
        for msg in chain[first_resolution_idx + 1:]:
            if msg['inbound'] and is_confirmed_resolved(msg['text']):
                confirmed = True
                break

        if not confirmed:
            continue

        resolved_count += 1

        # Build the conversation flow
        flow = []
        for msg in chain:
            role = 'customer' if msg['inbound'] else 'agent'
            text = msg['text'].strip()
            # Clean @mentions for readability
            text = re.sub(r'@\S+\s*', '', text).strip()
            if text:
                flow.append({
                    'role': role,
                    'text': text,
                })

        resolved_conversations[intent].append({
            'conversation_id': tid,
            'intent': intent,
            'message_count': len(flow),
            'flow': flow,
        })

    print(f"\n--- Extraction Results ---")
    print(f"  Total multi-turn English conversations: {total_convs:,}")
    print(f"  With actual resolution steps:           {has_resolution_steps:,}")
    print(f"  Customer confirmed resolved:            {resolved_count:,}")

    # Per-intent breakdown
    print(f"\n  {'Intent':<25} {'Resolved Convs':>15}")
    print(f"  {'-' * 42}")
    total_resolved = 0
    for intent in sorted(resolved_conversations):
        count = len(resolved_conversations[intent])
        total_resolved += count
        print(f"  {intent:<25} {count:>15}")
    print(f"  {'-' * 42}")
    print(f"  {'TOTAL':<25} {total_resolved:>15}")

    # For each intent, pick the best conversations (diverse, clear, multi-turn)
    curated = {}
    for intent, convs in resolved_conversations.items():
        # Sort by message count (prefer longer diagnostic flows) but cap at 10 msgs
        good = [c for c in convs if 3 <= c['message_count'] <= 10]
        good.sort(key=lambda x: x['message_count'], reverse=True)

        # Deduplicate — skip if agent text is very similar to one already selected
        selected = []
        seen_agent_texts = set()
        for conv in good:
            agent_texts = tuple(
                m['text'][:80] for m in conv['flow'] if m['role'] == 'agent'
            )
            # Simple dedup: skip if the first agent response is nearly identical
            first_agent = agent_texts[0] if agent_texts else ''
            if first_agent in seen_agent_texts:
                continue
            seen_agent_texts.add(first_agent)
            selected.append(conv)
            if len(selected) >= 25:
                break

        curated[intent] = selected

    # Save
    output_path = os.path.join(KB_DIR, 'historical_responses.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(curated, f, indent=2, ensure_ascii=False)

    print(f"\n  Curated per intent:")
    total_curated = 0
    for intent in sorted(curated):
        count = len(curated[intent])
        total_curated += count
        print(f"    {intent:<25} {count:>5} conversations")
    print(f"    {'TOTAL':<25} {total_curated:>5}")
    print(f"\n  Saved → {output_path}")

    # Print 2 sample flows
    print(f"\n{'=' * 65}")
    print(f"  SAMPLE RESOLVED CONVERSATIONS")
    print(f"{'=' * 65}")

    shown = 0
    for intent in ['playback_issue', 'shuffle_queue', 'device_compatibility',
                    'subscription_billing', 'download_offline', 'account_access']:
        if intent in curated and curated[intent]:
            conv = curated[intent][0]
            print(f"\n  --- [{intent}] ({conv['message_count']} messages) ---")
            for msg in conv['flow']:
                role = 'CUST' if msg['role'] == 'customer' else 'SPOT'
                print(f"    [{role}] {msg['text'][:150]}")
            shown += 1
            if shown >= 4:
                break


if __name__ == '__main__':
    main()
