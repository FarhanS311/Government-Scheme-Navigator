# Step 6: RAGAS Evaluation

**Status:** done
**Date:** 2026-09-14

## Requirement

Evaluate the RAG pipeline on a golden question set with RAGAS metrics per query. Log results to JSONL and print a summary metrics table from a batch CLI.

## Acceptance criteria

- [x] Golden set with 8–15 question/`ground_truth` pairs committed
- [x] RAGAS: faithfulness, answer_relevancy, context_precision per query
- [x] JSONL log with timestamp, question, answer preview, metrics
- [x] `run_eval` runs end-to-end unattended; prints metrics table
- [x] `RAGAnswer.contexts` populated from reranked chunk texts
- [x] Tests pass

## Implementation notes

- `app.evaluation` — `EvalRecord`, `load_golden_set`, `evaluate_single`, `log_eval_record`, `format_metrics_table`
- `scripts.run_eval` — batch loop: golden set → `ask_question` → RAGAS → JSONL → stdout table
- `data/eval/golden_set.json` — 12 pairs aligned with test scheme corpus
- `data/logs/ragas.jsonl` — append-only eval log (gitignored)
- `langchain-community==0.3.31` pinned for ragas 0.4.3 VertexAI import compatibility

## Manual checklist

```bash
cd backend
source venv/bin/activate
python -m scripts.build_index /path/to/scheme.pdf   # if index missing
python -m scripts.run_eval
# verify: metrics table printed, data/logs/ragas.jsonl has 12 lines
```

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_evaluation.py -v
```

## Follow-ups

- FastAPI endpoints
- Streamlit UI
- Optional `--eval` flag on `ask_rag`
