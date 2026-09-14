# Government Scheme Navigator

Single-document RAG assistant: upload one public scheme PDF, ask eligibility and application-process questions, and get answers grounded in that document with cited source chunks.

This repository is built step by step. **Step 7 (current):** Streamlit UI + FastAPI endpoints.

## Layout

- `backend/` — FastAPI + LangChain pipeline (ingestion lives here today)
- `frontend/` — Streamlit UI calling the FastAPI backend

## Setup

Python 3.12 virtual environments:

```bash
# Backend (reuse if already created)
python3 -m venv backend/venv
backend/venv/bin/pip install -r backend/requirements.txt

# Frontend
python3 -m venv frontend/venv
frontend/venv/bin/pip install -r frontend/requirements.txt
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

## Step 5: Cross-encoder re-ranking

After FAISS retrieves top-N candidates (`N=20`), `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores chunks against the user question. Top `k` re-ranked chunks go to the LLM.

```bash
python -m scripts.ask_rag "What documents are needed for the application process?"
```

CLI prints before/after rank table and appends JSONL logs to `backend/data/logs/rerank.jsonl`.

Example where re-ranking changes rank-1 (bi-encoder favored eligibility chunk; cross-encoder promoted application chunk):

See [docs/rerank-before-after.txt](docs/rerank-before-after.txt).

First cross-encoder run downloads model (~100MB).

## Step 6: RAGAS evaluation

Batch-evaluate the RAG pipeline against a committed golden set (`backend/data/eval/golden_set.json`, 12 labeled question/`ground_truth` pairs). For each question, `run_eval` calls the full `ask_question` pipeline, scores with RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`), appends JSONL logs, and prints a metrics table.

Build the FAISS index from the same scheme PDF used in tests before running eval:

```bash
cd backend
source venv/bin/activate
python -m scripts.build_index /path/to/scheme.pdf   # once
python -m scripts.run_eval
python -m scripts.run_eval --golden-set data/eval/golden_set.json --index-dir data/index
```

Example output:

```
| id               | faithfulness | answer_relevancy | context_precision |
|------------------|--------------|------------------|-------------------|
| eligibility-01   | 0.91         | 0.87             | 0.80              |
| MEAN             | 0.85         | 0.82             | 0.78              |

Logged 12 evaluation(s) to data/logs/ragas.jsonl
```

RAGAS uses OpenRouter (same LLM as the RAG chain). Each golden-set question triggers several LLM calls for scoring.

## Step 7: Frontend and source attribution UI

FastAPI exposes the pipeline to a Streamlit UI. Upload a PDF in the browser, ask questions, see answers with an expandable **Sources** panel (page numbers + chunk text).

### Backend API

```bash
cd backend
source venv/bin/activate
cp .env.example .env   # set OPENROUTER_API_KEY
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /api/status` — whether a document is indexed
- `POST /api/upload` — multipart PDF upload; ingests and builds FAISS index
- `POST /api/ask` — JSON `{ "question": "...", "k": 4 }`; returns answer + sources

### Streamlit UI

In a second terminal:

```bash
cd frontend
source venv/bin/activate
streamlit run app.py
```

Opens at `http://localhost:8501`. Optional env: `BACKEND_URL=http://127.0.0.1:8000` (default).

**Reviewer checklist:** upload scheme PDF, ask an eligibility or application question, confirm answer cites the document and **Sources** expanders show page + chunk text.

First upload may be slow while embedding (~90MB) and reranker (~100MB) models download.

### Tests

```bash
cd backend
source venv/bin/activate
pytest
```
