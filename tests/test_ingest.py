"""Tests for the ingestion pipeline."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import chunk_text, chunk_documents


def test_chunk_text_splits_long_text():
    long_text = "word " * 600
    chunks = chunk_text(long_text, "test.txt", 1)
    assert len(chunks) > 1


def test_chunk_text_preserves_metadata():
    chunks = chunk_text("Hello world " * 10, "test.pdf", 3)
    for chunk in chunks:
        assert chunk["source"] == "test.pdf"
        assert chunk["page"] == 3


def test_chunk_documents_empty_input():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_text_short_text():
    chunks = chunk_text("Short text", "doc.txt", 1)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text"
