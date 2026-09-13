"""
Script 5: Build the RAG retrieval pipeline

Chunks KB articles, embeds them using sentence-transformers,
stores in ChromaDB for semantic retrieval.

Outputs:
  - knowledge_base/chroma_db/     (ChromaDB persistent store)
  - knowledge_base/chunks.json    (all KB chunks with metadata)

"""

import json
import os

from kb_retriever import KBRetriever

KB_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')


def flatten_value(value, prefix=""):
    """Recursively flatten a nested dict/list into readable text lines."""
    lines = []
    if isinstance(value, str):
        lines.append(f"{prefix}{value}")
    elif isinstance(value, list):
        for item in value:
            lines.extend(flatten_value(item, prefix))
    elif isinstance(value, dict):
        for k, v in value.items():
            label = k.replace('_', ' ').title()
            lines.extend(flatten_value(v, f"{label}: "))
    return lines


def chunk_kb(kb):
    """Break KB into retrievable chunks, one per article/topic."""
    chunks = []
    for intent, data in kb['intents'].items():
        for article in data.get('articles', []):
            text_parts = []
            topic = article.get('topic', intent)
            text_parts.append(f"Topic: {topic}")
            text_parts.append(f"Intent: {intent}")

            for key, value in article.items():
                if key in ('topic', 'url'):
                    continue
                lines = flatten_value(value)
                if lines:
                    label = key.replace('_', ' ').title()
                    text_parts.append(f"\n{label}:")
                    text_parts.extend(f"  - {line}" for line in lines)

            chunk_text = '\n'.join(text_parts)
            chunks.append({
                'intent': intent,
                'topic': topic,
                'url': article.get('url', ''),
                'text': chunk_text,
            })

    return chunks


def main():
    # Load KB
    kb_path = os.path.join(KB_DIR, 'kb.json')
    with open(kb_path, 'r') as f:
        kb = json.load(f)

    # Chunk it
    chunks = chunk_kb(kb)
    print(f"Created {len(chunks)} KB chunks across {len(kb['intents'])} intents\n")

    for intent in sorted(kb['intents']):
        intent_chunks = [c for c in chunks if c['intent'] == intent]
        topics = [c['topic'] for c in intent_chunks]
        print(f"  {intent:<25} {len(intent_chunks)} chunk(s): {', '.join(topics)}")

    # Build retriever with embeddings + ChromaDB
    print(f"\nBuilding semantic retriever (sentence-transformers + ChromaDB)...")
    retriever = KBRetriever(chunks)

    # Save chunks as JSON for reference
    chunks_path = os.path.join(KB_DIR, 'chunks.json')
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved chunks -> {chunks_path}")

    # Test retrieval — includes tricky paraphrased queries
    test_queries = [
        # Direct keyword matches (TF-IDF would get these too)
        ("my bluetooth won't connect to my car", "device_compatibility"),
        ("I got charged twice for premium", "subscription_billing"),
        ("shuffle keeps playing same songs", "shuffle_queue"),
        ("someone hacked my spotify account", "account_access"),
        ("too many ads I hate it", "ads_complaints"),
        ("app keeps crashing on my iphone", "playback_issue"),

        # Paraphrased — NO keywords from the KB articles (TF-IDF would FAIL these)
        ("wireless speaker won't pair with my phone", "device_compatibility"),
        ("took money from my bank but I cancelled", "subscription_billing"),
        ("it keeps repeating the same tracks over and over", "shuffle_queue"),
        ("the songs sound muffled and distorted", "audio_quality"),
        ("this artist used to be here but all their stuff vanished", "content_availability"),
        ("my saved music for the plane is all gone", "download_offline"),
    ]

    print(f"\n--- Retrieval Test ---\n")
    correct = 0
    for query, expected_intent in test_queries:
        results = retriever.retrieve(query, intent=expected_intent, top_k=2)
        top = results[0]
        match = top['intent'] == expected_intent
        if match:
            correct += 1
        status = "✓" if match else "✗"
        print(f"  {status} \"{query}\"")
        print(f"    → [{top['score']:.3f}] {top['topic']} ({top['intent']})")
        if len(results) > 1:
            r2 = results[1]
            print(f"    → [{r2['score']:.3f}] {r2['topic']} ({r2['intent']})")
        print()

    print(f"  Retrieval accuracy: {correct}/{len(test_queries)} ({correct/len(test_queries):.0%})")


if __name__ == '__main__':
    main()
