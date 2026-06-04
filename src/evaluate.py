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


def evaluate(qa_path: Path = QA_PATH) -> dict:
    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results = []
    hit_count = 0
    total_overlap = 0.0

    print(f"Running evaluation on {len(qa_pairs)} Q&A pairs...\n")
    print(f"{'#':<4} {'Hit':<6} {'Overlap':<10} Question")
    print("-" * 70)

    for i, qa in enumerate(qa_pairs, 1):
        question = qa["question"]
        expected_source = qa["expected_source"]
        expected_keywords = qa["expected_keywords"]

        hit = retrieval_hit(question, expected_source)
        result = answer(question)
        overlap = keyword_overlap(result["answer"], expected_keywords)

        hit_count += int(hit)
        total_overlap += overlap

        results.append({
            "question": question,
            "retrieval_hit": hit,
            "keyword_overlap": round(overlap, 2),
            "answer_snippet": result["answer"][:120],
        })

        print(f"{i:<4} {'HIT' if hit else 'MISS':<6} {overlap:.0%}{'':>4} {question[:55]}")

    n = len(qa_pairs)
    retrieval_hit_rate = hit_count / n
    avg_overlap = total_overlap / n

    print("\n" + "=" * 70)
    print(f"  Retrieval hit-rate  : {retrieval_hit_rate:.0%}  ({hit_count}/{n} questions)")
    print(f"  Avg keyword overlap : {avg_overlap:.0%}")
    print("=" * 70)

    return {
        "retrieval_hit_rate": round(retrieval_hit_rate, 3),
        "avg_keyword_overlap": round(avg_overlap, 3),
        "n_questions": n,
        "results": results,
    }


if __name__ == "__main__":
    evaluate()
