"""
Script 8: Embed historical responses into ChromaDB

Creates a second collection 'spotify_historical' alongside the KB collection.
At query time, the bot retrieves from both — KB for accuracy, historical for tone.

"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from sentence_transformers import SentenceTransformer

KB_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
CHROMA_DIR = os.path.join(KB_DIR, 'chroma_db')


def main():
    # Load historical responses
    hist_path = os.path.join(KB_DIR, 'historical_responses.json')
    with open(hist_path, 'r') as f:
        historical = json.load(f)

    # Build documents: each document = one full conversation as readable text
    documents = []
    metadatas = []
    ids = []
    idx = 0

    for intent, convs in historical.items():
        for conv in convs:
            # Build readable conversation text
            lines = [f"Intent: {intent}"]
            for msg in conv['flow']:
                role = "Customer" if msg['role'] == 'customer' else "Agent"
                lines.append(f"{role}: {msg['text']}")
            full_text = '\n'.join(lines)

            # Also extract just the customer's first message for similarity matching
            first_customer = next(
                (m['text'] for m in conv['flow'] if m['role'] == 'customer'), ''
            )

            documents.append(full_text)
            metadatas.append({
                'intent': intent,
                'first_customer_msg': first_customer[:200],
                'message_count': conv['message_count'],
                'conversation_id': conv.get('conversation_id', ''),
            })
            ids.append(f"hist_{idx}")
            idx += 1

    print(f"Prepared {len(documents)} historical conversations for embedding")
    for intent in sorted(historical):
        print(f"  {intent:<25} {len(historical[intent]):>5}")

    # Load embedding model
    print(f"\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed all documents
    print(f"Embedding {len(documents)} conversations...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # Store in ChromaDB
    print(f"Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if present
    try:
        client.delete_collection("spotify_historical")
    except Exception:
        pass

    collection = client.create_collection(
        name="spotify_historical",
        metadata={"hnsw:space": "cosine"},
    )

    # Add in batches (ChromaDB has limits)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            ids=ids[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"  Indexed {len(documents)} conversations into ChromaDB collection 'spotify_historical'")

    # Test retrieval
    test_queries = [
        ("my app keeps crashing", "playback_issue"),
        ("shuffle plays same songs over and over", "shuffle_queue"),
        ("bluetooth speaker wont connect", "device_compatibility"),
        ("charged twice for premium", "subscription_billing"),
        ("my downloads disappeared", "download_offline"),
        ("someone is using my account", "account_access"),
    ]

    print(f"\n--- Retrieval Test ---\n")
    for query, expected in test_queries:
        results = collection.query(
            query_embeddings=model.encode([query]).tolist(),
            n_results=2,
            where={"intent": expected},
        )
        print(f'  "{query}"')
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                dist = results['distances'][0][i]
                score = 1 - dist
                # Show first customer and agent lines
                lines = doc.split('\n')
                cust_line = next((l for l in lines if l.startswith('Customer:')), '')
                agent_line = next((l for l in lines if l.startswith('Agent:')), '')
                print(f"    [{score:.3f}] {cust_line[:80]}")
                print(f"             {agent_line[:80]}")
        print()


if __name__ == '__main__':
    main()
