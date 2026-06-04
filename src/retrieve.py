"""
Retrieval: embed a user query and return the top-k most relevant chunks from ChromaDB.
"""

import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K_RESULTS", 5))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

_model = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Return top-k chunks most relevant to the query, with source metadata."""
    model = _get_model()
    query_embedding = model.encode([query]).tolist()
    collection = _get_collection()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "Unknown"),
            "page": results["metadatas"][0][i].get("page", ""),
            "distance": results["distances"][0][i],
        })
    return chunks
