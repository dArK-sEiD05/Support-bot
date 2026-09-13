"""Shared KB retriever — embeddings via sentence-transformers + ChromaDB."""

import os
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base', 'chroma_db')


class KBRetriever:
    def __init__(self, chunks=None, load_existing=False):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)

        if load_existing:
            self.collection = self.client.get_collection("spotify_kb")
            self.chunks = None  # not needed for querying
        elif chunks:
            self.chunks = chunks
            # Delete if exists, recreate fresh
            try:
                self.client.delete_collection("spotify_kb")
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name="spotify_kb",
                metadata={"hnsw:space": "cosine"},
            )
            self._index_chunks(chunks)
        else:
            raise ValueError("Provide chunks to index or set load_existing=True")

    def _index_chunks(self, chunks):
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"intent": c['intent'], "topic": c['topic'], "url": c.get('url', '')}
                     for c in chunks]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        print(f"  Indexed {len(chunks)} chunks into ChromaDB")

    def retrieve(self, query, intent=None, top_k=3):
        query_embedding = self.model.encode([query]).tolist()

        # If intent provided, try intent-filtered search first
        if intent:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where={"intent": intent},
            )
            # If we got results from the matching intent, use them
            # but also get unfiltered results to compare
            unfiltered = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )
            # Merge: intent-matched results first, then fill with unfiltered
            seen = set()
            merged_results = []
            for source in [results, unfiltered]:
                if source['documents'] and source['documents'][0]:
                    for i, doc in enumerate(source['documents'][0]):
                        doc_id = source['ids'][0][i]
                        if doc_id not in seen:
                            seen.add(doc_id)
                            dist = source['distances'][0][i] if source['distances'] else 0
                            score = 1 - dist  # cosine distance to similarity
                            merged_results.append({
                                'topic': source['metadatas'][0][i]['topic'],
                                'intent': source['metadatas'][0][i]['intent'],
                                'url': source['metadatas'][0][i].get('url', ''),
                                'score': round(score, 4),
                                'text': doc,
                            })
            return merged_results[:top_k]
        else:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
            )
            output = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    dist = results['distances'][0][i] if results['distances'] else 0
                    score = 1 - dist
                    output.append({
                        'topic': results['metadatas'][0][i]['topic'],
                        'intent': results['metadatas'][0][i]['intent'],
                        'url': results['metadatas'][0][i].get('url', ''),
                        'score': round(score, 4),
                        'text': doc,
                    })
            return output
