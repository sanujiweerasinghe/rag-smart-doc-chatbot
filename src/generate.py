import os
from dotenv import load_dotenv
import ollama
from retrieve import retrieve

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY using information from the context below.
- If the context does not contain enough information to answer, say: "I could not find this in the provided documents."
- Always cite the source document(s) you used at the end of your answer.
- Be concise and accurate."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_text = ""
    for i, chunk in enumerate(chunks, 1):
        source = os.path.basename(chunk["source"])
        page = chunk.get("page", "")
        label = f"{source}, page {page}" if page else source
        context_text += f"\n[{i}] ({label}):\n{chunk['text']}\n"

    return f"""{SYSTEM_PROMPT}

Context:
{context_text}

Question: {query}

Answer:"""
