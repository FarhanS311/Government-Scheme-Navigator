# Step 3: LCEL RAG Chain

**Status:** done
**Date:** 2026-09-14

## Requirement

Connect Step 2 retrieval to an LLM via LCEL: retrieve chunks, format context, prompt, generate answer. Return structured output with answer text and source citations. Refuse when context is insufficient.

## Acceptance criteria

- [x] LCEL chain with explicit retrieve → prompt → LLM stages
- [x] Context-only prompt; explicit refusal phrase when answer missing
- [x] `RAGAnswer` with `answer` + `sources` (`id`, `page`, `snippet`)
- [x] Sources from retrieval step, not LLM hallucination
- [x] Mock LLM tests for in-scope and out-of-scope questions
- [x] OpenRouter config via `.env` (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL`)

## Implementation notes

- `app.config.get_llm()` — `ChatOpenAI` pointed at OpenRouter
- `app.rag_chain` — `create_rag_chain`, `ask_question`, `RAGAnswer`, `SourceCitation`
- Chain: `RunnableLambda(retrieve)` → `RunnablePassthrough.assign(answer_text=prompt|llm)` → `RAGAnswer`
- CLI: `python -m scripts.ask_rag "<question>"`
- Default retrieval `k=4`

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_rag_chain.py -v
```

Manual live check (needs `OPENROUTER_API_KEY` in `backend/.env`):

```bash
python -m scripts.build_index /tmp/scheme-sample.pdf
python -m scripts.ask_rag "Who is eligible?"
python -m scripts.ask_rag "What is the capital of France?"
```

## Follow-ups

- Cross-encoder re-ranking
- RAGAS evaluation per query
- FastAPI endpoints
- Streamlit UI
