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
