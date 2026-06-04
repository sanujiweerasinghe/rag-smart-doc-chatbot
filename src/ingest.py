import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for file in data_dir.iterdir():
        if file.suffix.lower() == ".pdf":
            reader = PdfReader(str(file))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "source": str(file), "page": i + 1})
        elif file.suffix.lower() in (".md", ".txt"):
            text = file.read_text(encoding="utf-8")
            if text.strip():
                docs.append({"text": text, "source": str(file), "page": 1})
    return docs


def chunk_text(text: str, source: str, page: int) -> list[dict]:
    words = text.split()
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk_words = words[i: i + CHUNK_SIZE]
        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words),
                "source": source,
                "page": page,
            })
    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["source"], doc["page"]))
    return chunks


def build_vector_store(chunks: list[dict]) -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )


def ingest(data_dir: Path = DATA_DIR) -> None:
    print(f"Loading documents from {data_dir}...")
    docs = load_documents(data_dir)
    if not docs:
        print("No documents found. Add PDFs or .md/.txt files to the data/ folder.")
        return

    print(f"Loaded {len(docs)} pages/documents. Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Embedding and storing in ChromaDB...")
    build_vector_store(chunks)
    print(f"Done. Vector store saved to {CHROMA_DB_PATH}")
    print(f"\nChunk size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} - why these numbers:")
    print("  - 500 words (~1-2 paragraphs): enough context per chunk, not too long.")
    print("  - 50-word overlap: prevents answers being cut off at chunk boundaries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    ingest(args.data_dir)
