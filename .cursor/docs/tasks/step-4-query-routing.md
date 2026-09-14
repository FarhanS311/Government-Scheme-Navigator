# Step 4: Query Routing / Decomposition

**Status:** done
**Date:** 2026-09-14

## Requirement

Add lightweight query routing before retrieval. Classify single-fact, multi-part, or summarization. Decompose multi-part questions into sub-queries, retrieve each, merge context, then generate answer.

## Acceptance criteria

- [x] Router classifies `single_fact`, `multi_part`, `summarization`
- [x] One LLM call returns JSON `sub_queries`
- [x] Per sub-query retrieval with deduped merge
- [x] `RAGAnswer` includes `query_type` and `sub_queries`
- [x] Multi-part test shows both chunk groups retrieved
- [x] Tests pass

## Implementation notes

- `app.query_router` — `RouteResult`, `route_query` via `with_structured_output`
- `app.rag_chain` — `_retrieve_multi`, `_merge_scored_chunks`, extended `RAGAnswer`
- Fallback on router failure: `single_fact` + original question
- CLI prints routing metadata

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_query_router.py tests/test_rag_chain.py -v
```

Manual (needs `OPENROUTER_API_KEY`):

```bash
python -m scripts.ask_rag "Who is eligible and how do I apply?"
```

## Follow-ups

- Cross-encoder re-ranking
- RAGAS evaluation
- FastAPI + Streamlit UI
