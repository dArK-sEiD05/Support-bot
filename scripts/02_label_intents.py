"""
Script 2: Label intents on customer tweets

Reads data/customer_tweets.json and labels each with an intent category.
Uses keyword matching as a first pass. Outputs:
  - data/labeled_tweets.json   (all tweets with intent labels)
  - data/intent_summary.json   (counts + samples per intent)

"""

import json
import os
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

INTENT_PATTERNS = {
    'playback_issue': [
        'not playing', "won't play", 'stops playing', 'keeps pausing',
        'keeps stopping', "can't play", 'no sound', "won't load",
        'buffering', 'skipping songs', 'crashes', 'freezes', 'frozen',
        'not working', 'app crashes', 'keep crashing', 'stopped working',
        'music stops', 'pauses randomly', 'cuts out', 'glitch',
        'bug', 'black screen', 'spinning', 'loading forever',
    ],
    'account_access': [
        "can't log in", "can't login", "can't sign in", 'locked out',
        'password', 'reset password', 'forgot password', "can't access",
        'account hacked', 'hacked', 'someone else', 'unauthorized',
        'stolen account', 'compromised', 'logged out', 'kicked out',
        "can't get in", 'wrong password', 'email changed',
    ],
    'subscription_billing': [
        'charged', 'billing', 'payment', 'refund', 'subscription',
        'cancel', 'premium', 'free trial', 'student discount',
        'family plan', 'double charged', 'still charging', 'money',
        'price', 'invoice', 'receipt', 'renew', 'upgrade',
        'downgrade', 'plan', 'coupon', 'promo', 'offer',
        'credit card', 'paypal', 'gift card', 'hulu bundle',
    ],
    'content_availability': [
        'not available', 'song missing', 'album missing', 'removed',
        "can't find song", "can't find album", 'where is',
        'bring back', 'no longer available', 'taken down',
        'greyed out', 'grayed out', 'not on spotify', 'when will',
        'licensing', 'unavailable', 'disappeared from',
    ],
    'download_offline': [
        'download', 'offline', 'downloaded songs', 'downloads disappeared',
        "can't download", 'offline mode', 'storage space',
        'downloads removed', 'lost downloads', 'offline device',
        'download limit', 'sync', 'pending download',
    ],
    'shuffle_queue': [
        'shuffle', 'queue', 'repeat', 'play in order', 'random order',
        "won't stop shuffling", 'shuffle off', 'clear queue',
        'play next', 'up next', 'shuffle sucks', 'not random',
        'same songs', 'shuffle broken', 'autoplay',
    ],
    'device_compatibility': [
        'bluetooth', 'chromecast', 'alexa', 'echo', 'apple watch',
        'car', 'roku', 'ps4', 'ps5', 'playstation', 'xbox',
        'smart tv', 'google home', 'fire stick', 'sonos',
        'connect', 'airplay', 'fitbit', 'garmin', 'samsung',
        'wear os', 'android auto', 'carplay', 'speaker',
    ],
    'feature_request': [
        'wish you', 'would be nice', 'please add', 'can you add',
        'suggestion', 'idea', 'why can\'t we', 'need a way to',
        'it would be great', 'you should add', 'when will you',
        'consider adding', 'vote for',
    ],
    'audio_quality': [
        'sound quality', 'audio quality', 'low quality', 'bitrate',
        'sounds bad', 'sounds terrible', 'distorted', 'hi-fi',
        'lossless', 'equalizer', 'eq', 'bass', 'volume low',
        'too quiet', 'too loud', 'volume normalization', '320',
    ],
    'playlist_library': [
        'playlist', 'library', 'discover weekly', 'daily mix',
        'release radar', 'recommended', 'songs disappeared',
        'saved songs', 'my music', 'liked songs', 'top songs',
        'wrapped', 'blend', 'collaborative', 'folder',
    ],
    'ads_complaints': [
        'ads', 'advert', 'advertisement', 'too many ads',
        'annoying ads', 'stop ads', 'ad free', 'ad-free',
        'commercial', '30 minutes', 'ad every', 'sick of ads',
    ],
    'country_region': [
        'not available in my country', 'when in india',
        'launch in', 'available in', 'not in my country',
        'region', 'my country', 'vpn',
    ],
}


def classify_tweet(text):
    text_lower = text.lower()
    scores = {}
    for intent, keywords in INTENT_PATTERNS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[intent] = matches

    if not scores:
        return 'other'

    return max(scores, key=scores.get)


def main():
    cust_path = os.path.join(DATA_DIR, 'customer_tweets.json')
    with open(cust_path, 'r', encoding='utf-8') as f:
        customer_tweets = json.load(f)

    print(f"Loaded {len(customer_tweets):,} customer tweets")

    # Label each tweet
    intent_counts = Counter()
    intent_samples = defaultdict(list)
    labeled = []

    for tweet in customer_tweets:
        intent = classify_tweet(tweet['text'])
        tweet['intent'] = intent
        labeled.append(tweet)

        intent_counts[intent] += 1
        if len(intent_samples[intent]) < 5:
            intent_samples[intent].append(tweet['text'][:150])

    # Print summary
    total = len(labeled)
    print(f"\n--- Intent Distribution ---\n")
    print(f"  {'Intent':<25} {'Count':>8} {'%':>7}")
    print(f"  {'-'*42}")
    for intent, count in intent_counts.most_common():
        pct = count / total * 100
        print(f"  {intent:<25} {count:>8,} {pct:>6.1f}%")
    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<25} {total:>8,}")

    labeled_excl_other = sum(c for i, c in intent_counts.items() if i != 'other')
    print(f"\n  Labeled (excl other):   {labeled_excl_other:,} ({labeled_excl_other/total*100:.1f}%)")
    print(f"  Unlabeled (other):      {intent_counts['other']:,} ({intent_counts['other']/total*100:.1f}%)")

    # Save labeled tweets
    labeled_path = os.path.join(DATA_DIR, 'labeled_tweets.json')
    with open(labeled_path, 'w', encoding='utf-8') as f:
        json.dump(labeled, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved labeled tweets   -> {labeled_path}")

    # Save intent summary
    summary = {}
    for intent in intent_counts:
        summary[intent] = {
            'count': intent_counts[intent],
            'percentage': round(intent_counts[intent] / total * 100, 2),
            'samples': intent_samples[intent],
        }

    summary_path = os.path.join(DATA_DIR, 'intent_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved intent summary   -> {summary_path}")


if __name__ == '__main__':
    main()
