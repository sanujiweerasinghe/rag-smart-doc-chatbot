# RAG Smart Doc Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents — with citations, not hallucinations.

## What it does

Upload your PDFs or text files, ask questions, and get accurate answers grounded in your documents. Every answer includes the exact source chunk it came from.

## Architecture

```
Your Documents (PDF / Markdown / TXT)
        ↓
   [ingest.py]  — chunk → embed → store in ChromaDB
        ↓
   User Question
        ↓
   [retrieve.py] — embed query → find top-5 similar chunks
        ↓
   [generate.py] — context + question → Ollama (llama3.2) → Answer + Citations
        ↓
   [app.py]      — Streamlit chat UI
```

## Tech Stack

| Tool | Role |
|------|------|
| `sentence-transformers` | Local text embeddings (`all-MiniLM-L6-v2`) |
| `ChromaDB` | Local vector database |
| `Ollama (llama3.2)` | Local LLM — free, no API cost |
| `pypdf` | PDF text extraction |
| `Streamlit` | Chat UI |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull the model

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2
```

### 3. Add your documents

Place your PDF, `.md`, or `.txt` files in the `data/` folder.

### 4. Ingest documents

```bash
python src/ingest.py
```

### 5. Run the chatbot

```bash
streamlit run src/app.py
```

## Evaluation Results

Evaluated on 15 Q&A pairs covering all 5 document topics (Simple/Double/Triple Exponential Smoothing, ARCH, GARCH).

| Metric | Score |
|--------|-------|
| Retrieval hit-rate | **100%** (15/15) |
| Avg keyword overlap | **66%** |

Run the evaluation yourself:

```bash
python src/evaluate.py
```

## Run Tests

```bash
pytest tests/
```

## Key Design Decisions

- **Chunk size 500 tokens, overlap 50** — balances retrieval precision vs. context completeness
- **`all-MiniLM-L6-v2` embeddings** — fast, lightweight, strong semantic similarity
- **Grounded generation** — the LLM is instructed to answer *only* from retrieved context, preventing hallucination
- **Fully local** — no API keys required; runs entirely on your machine

## What I'd improve / how I'd scale

- Upgrade to RAGAS-based evaluation for deeper answer faithfulness scoring
- Swap ChromaDB for pgvector for production-scale deployments
- Add re-ranking (cross-encoder) to improve retrieval quality
- Support for multi-turn conversation memory
