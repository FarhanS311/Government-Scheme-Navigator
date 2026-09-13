# Step 1: Ingestion and Chunking

**Status:** done
**Date:** 2026-09-13

## Requirement

Load a single PDF, extract text per page (preserving page numbers), recursively chunk it, and attach source filename, page number, and chunk index so later RAG citations can point at the right place.

## Acceptance criteria

- [x] Uploading/loading a PDF produces a list of chunk objects
- [x] Page numbers are 1-based and match the source page
- [x] Metadata includes source filename, page number, and chunk index
- [x] Chunks are printed to the console via a CLI
- [x] Empty pages are skipped
- [x] Missing files and non-PDF paths raise clear errors

## Implementation notes

- Loader: LangChain `PyPDFLoader` (per-page `Document`s)
- Splitter: `RecursiveCharacterTextSplitter` with `CHUNK_SIZE=2500` and `CHUNK_OVERLAP=300` (~600 tokens, ~12% overlap)
- Split per page (do not concatenate the full PDF first) so chunks never span pages
- Contract: `app.ingestion.Chunk` and `ingest_pdf(path) -> list[Chunk]`
- CLI: `python -m scripts.ingest_pdf <pdf>`
- Frontend venv created; no UI code in this step

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_ingestion.py -v
```

Cases: missing file, non-PDF, page numbers + source + sequential `chunk_index`, empty page skipped, overlap on a long page.

## Follow-ups

- FastAPI upload endpoint
- Embeddings + FAISS
- LCEL RAG chain, re-ranking, RAGAS, Streamlit UI
