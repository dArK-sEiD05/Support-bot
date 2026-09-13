"""
Script 1: Extract all SpotifyCares conversations from twcs.csv

Reads the raw CSV, builds conversation chains, and outputs:
  - data/conversations.json  (all threaded chats)
  - data/customer_tweets.json (just customer first-messages for intent labeling)

"""

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'twcs.csv')


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


def is_english(text):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.8


def main():
    print("Loading tweets from CSV...")
    tweets = load_tweets(CSV_PATH)
    print(f"  Loaded {len(tweets):,} tweets")

    # Find conversation starters: customer tweets with no parent that SpotifyCares replied to
    starters = []
    for tid, t in tweets.items():
        if t['inbound'] and not t['in_response_to_tweet_id']:
            resp_ids = [r.strip() for r in t['response_tweet_id'].split(',') if r.strip()]
            for rid in resp_ids:
                if rid in tweets and tweets[rid]['author_id'] == 'SpotifyCares':
                    starters.append(tid)
                    break

    print(f"  Found {len(starters):,} conversation starters")

    # Build conversations
    all_visited = set()
    conversations = []
    customer_tweets = []

    for tid in starters:
        if tid in all_visited:
            continue

        chain = follow_chain(tid, tweets, set())
        chain_ids = set(c['tweet_id'] for c in chain)

        # Skip if we already processed part of this chain
        if chain_ids & all_visited:
            continue
        all_visited.update(chain_ids)

        # Sort by timestamp
        chain.sort(key=lambda x: x['created_at'])

        spotify_count = sum(1 for c in chain if c['author_id'] == 'SpotifyCares')
        customer_count = sum(1 for c in chain if c['inbound'])
        english = all(is_english(c['text']) for c in chain)

        conv = {
            'conversation_id': tid,
            'message_count': len(chain),
            'spotify_messages': spotify_count,
            'customer_messages': customer_count,
            'is_english': english,
            'messages': chain,
        }
        conversations.append(conv)

        # Extract the first customer message for intent labeling
        first_customer = next((c for c in chain if c['inbound']), None)
        if first_customer:
            customer_tweets.append({
                'conversation_id': tid,
                'tweet_id': first_customer['tweet_id'],
                'text': first_customer['text'],
                'is_english': english,
                'conversation_length': len(chain),
            })

    # Stats
    total_msgs = sum(c['message_count'] for c in conversations)
    eng_convs = sum(1 for c in conversations if c['is_english'])
    multi_turn = sum(1 for c in conversations if c['message_count'] >= 3)

    print(f"\n--- Extraction Results ---")
    print(f"  Total conversations:    {len(conversations):,}")
    print(f"  Total messages:         {total_msgs:,}")
    print(f"  English conversations:  {eng_convs:,}")
    print(f"  Multi-turn (3+ msgs):   {multi_turn:,}")
    print(f"  Customer tweets:        {len(customer_tweets):,}")

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)

    conv_path = os.path.join(DATA_DIR, 'conversations.json')
    with open(conv_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved conversations    -> {conv_path}")

    cust_path = os.path.join(DATA_DIR, 'customer_tweets.json')
    with open(cust_path, 'w', encoding='utf-8') as f:
        json.dump(customer_tweets, f, indent=2, ensure_ascii=False)
    print(f"  Saved customer tweets  -> {cust_path}")


if __name__ == '__main__':
    main()
