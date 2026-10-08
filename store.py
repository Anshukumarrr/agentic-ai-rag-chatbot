"""
Vector store: a local, persistent ChromaDB collection.

The task allows "Pinecone (or any Vector DB of your choice)". Chroma is used
because it needs no account and no API key, so a reviewer can clone the repo
and run it immediately. This is the only file that knows which vector database
is in use - to move to Pinecone (or FAISS / Qdrant / pgvector) reimplement
build(), search() and count() and nothing else changes.
"""

import chromadb

COLLECTION = "agentic_ai_ebook"


def _client(path: str):
    return chromadb.PersistentClient(path=path)


def _create(client):
    # cosine distance so that 1 - distance is a similarity in [-1, 1]
    try:  # chromadb >= 0.6 / 1.x
        return client.create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    except TypeError:  # chromadb 0.5.x
        return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build(path, ids, embeddings, documents, metadatas, batch=256):
    """Drop the collection if it exists, recreate it, and insert every chunk."""
    client = _client(path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = _create(client)
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    return col.count()


def search(path, embedding, top_k=4):
    """Return the top_k most similar chunks as dicts with a similarity score."""
    col = _client(path).get_collection(COLLECTION)
    res = col.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "text": doc,
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "score": round(1.0 - float(dist), 4),
            }
        )
    return hits


def count(path):
    try:
        return _client(path).get_collection(COLLECTION).count()
    except Exception:
        return 0
