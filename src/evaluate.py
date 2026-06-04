import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from retrieve import retrieve
from generate import answer


QA_PATH = Path(__file__).parent.parent / "eval" / "qa_pairs.json"


def _normalize(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", s.lower())


def retrieval_hit(question: str, expected_source: str, top_k: int = 5) -> bool:
    """Return True if the expected source document appears in top-k retrieved chunks."""
    chunks = retrieve(question, top_k=top_k)
    needle = _normalize(expected_source)
    for chunk in chunks:
        if needle in _normalize(os.path.basename(chunk["source"])):
            return True
    return False


def keyword_overlap(answer_text: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords) if keywords else 0.0
