# Step 2: Embeddings and FAISS Index

**Status:** done
**Date:** 2026-09-14

## Requirement

Embed Step 1 chunks locally, build a persisted FAISS index, and expose `query_index(question, k)` returning top-k chunks with similarity scores. No API keys; reload from disk without re-embedding the corpus.

## Acceptance criteria

- [x] Chunks embedded with `sentence-transformers/all-MiniLM-L6-v2`
- [x] FAISS index persisted to `backend/data/index/` and reloadable
- [x] `VectorIndex.query_index(question, k)` returns `ScoredChunk` list with scores
- [x] Same question twice yields identical top-k `(chunk_index, score)` tuples
- [x] Different questions return different top chunks on discriminative corpus
- [x] Tests pass

## Implementation notes

- Module: `app.vector_store` — `ScoredChunk`, `build_index`, `load_index`, `VectorIndex`
- Embeddings: LangChain `HuggingFaceEmbeddings` with `normalize_embeddings=True`
- FAISS: `IndexFlatIP` via `DistanceStrategy.MAX_INNER_PRODUCT`
- Persistence: `data/index/index.faiss` + `index.pkl` (LangChain `save_local`)
- CLIs: `scripts/build_index.py`, `scripts/query_index.py`
- First run downloads MiniLM weights (~90MB)

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_vector_store.py -v
```

## Follow-ups

- LCEL RAG chain and answer generation
- Cross-encoder re-ranking
- RAGAS evaluation
- FastAPI upload + query endpoints
- Streamlit UI
