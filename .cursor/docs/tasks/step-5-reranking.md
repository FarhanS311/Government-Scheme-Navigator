# Step 5: Cross-Encoder Re-Ranking

**Status:** done
**Date:** 2026-09-14

## Requirement

Re-score top-N FAISS chunks with a cross-encoder before LLM generation. Log pre-rerank and post-rerank ordering. Prove at least one query where rank-1 changes.

## Acceptance criteria

- [x] `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks top-N candidates
- [x] Pre/post ordering logged (CLI table + JSONL)
- [x] Rank-1 change demonstrated in test and `docs/rerank-before-after.txt`
- [x] Tests pass

## Implementation notes

- `app.reranker` — `RerankEntry`, `rerank_chunks`, `format_rerank_comparison`, `log_rerank_to_jsonl`
- `RERANK_TOP_N=20`, final `k=4` to LLM
- Re-rank against original user question after multi-query merge
- `RAGAnswer.rerank_log` added

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_reranker.py -v
```

## Follow-ups

- RAGAS evaluation per query
- FastAPI endpoints
- Streamlit UI
