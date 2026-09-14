# Government Scheme Navigator

Single-document RAG assistant: upload one public scheme PDF, ask eligibility and application-process questions, and get answers grounded in that document with cited source chunks.

This repository is built step by step. **Step 4 (current):** query routing and decomposition.

## Layout

- `backend/` — FastAPI + LangChain pipeline (ingestion lives here today)
- `frontend/` — Streamlit/HTML UI (venv only for now)

## Setup

Python 3.12 virtual environments:

```bash
# Backend (reuse if already created)
python3 -m venv backend/venv
backend/venv/bin/pip install -r backend/requirements.txt

# Frontend (no app dependencies yet)
python3 -m venv frontend/venv
```

Activate the backend env:

```bash
source backend/venv/bin/activate
```

## Step 1: Ingestion and chunking

Load a PDF with LangChain `PyPDFLoader`, split each page with `RecursiveCharacterTextSplitter` (`chunk_size=2500`, `chunk_overlap=300`), and keep metadata: source filename, 1-based page number, chunk index.

Print chunks to the console:

```bash
cd backend
source venv/bin/activate
python -m scripts.ingest_pdf /path/to/scheme.pdf
```

Each printed block includes `chunk_index`, `source`, `page`, character length, and the chunk text.

## Step 2: Embeddings and FAISS index

Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` (no API key). Build a FAISS `IndexFlatIP` index with normalized vectors; persist to `backend/data/index/` so the corpus is not re-embedded on restart.

Build index from a PDF:

```bash
cd backend
source venv/bin/activate
python -m scripts.build_index /path/to/scheme.pdf
```

Query the persisted index:

```bash
python -m scripts.query_index "Who is eligible for this scheme?" -k 3
python -m scripts.query_index "How do I apply?" -k 3
```

The first embedding run downloads the MiniLM model (~90MB). Each result prints `score`, `chunk_index`, `source`, `page`, and a text preview.

## Step 3: LCEL RAG chain

Wire retrieval → prompt → LLM via LCEL. LLM uses OpenRouter (OpenAI-compatible API). Copy env template and add your key:

```bash
cd backend
cp .env.example .env
# set OPENROUTER_API_KEY in .env
```

Default model: `openai/gpt-4o-mini` (override with `OPENROUTER_MODEL`).

Ask a question (requires built index from Step 2):

```bash
python -m scripts.ask_rag "Who is eligible for this scheme?"
python -m scripts.ask_rag "What is the capital of France?"
```

Returns answer text plus source citations (`id`, `page`, `snippet`) from retrieved chunks. If the answer is not in the document, the model must respond with: `The answer is not in this document.`

## Step 4: Query routing / decomposition

Before retrieval, one router LLM call classifies the question as `single_fact`, `multi_part`, or `summarization` and returns JSON sub-queries. Multi-part questions split into separate searches; results merge (deduped by chunk id) before answer generation.

```bash
python -m scripts.ask_rag "Who is eligible and how do I apply?"
```

CLI now prints `query_type` and `sub_queries`. Multi-part answers should cite sources from multiple pages/chunks.

### Tests

```bash
cd backend
source venv/bin/activate
pytest
```
