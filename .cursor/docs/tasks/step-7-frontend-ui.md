# Step 7: Frontend and Source Attribution UI

**Status:** done
**Date:** 2026-09-14

## Requirement

Minimum demo UI: upload PDF, ask question, show answer with expandable sources panel (chunk text + page numbers). Fresh reviewer can upload and get cited answer without CLI.

## Acceptance criteria

- [x] FastAPI: `POST /api/upload`, `POST /api/ask`, `GET /api/status`
- [x] Streamlit: upload box, question input, answer display, expandable sources
- [x] Sources show page number, chunk id, full chunk text
- [x] CORS for Streamlit on port 8501
- [x] API tests with upload fixture + mocked `ask_question`
- [x] README Step 7 run instructions

## Implementation notes

- `app/schemas.py` — request/response models
- `app/document_store.py` — save upload, build index, metadata, index cache
- `app/main.py` — FastAPI app + CORS
- `frontend/app.py` — Streamlit UI via `httpx`
- Upload replaces single-doc index (`data/index/`, `data/uploads/`)
- `sources[].text` from `RAGAnswer.contexts` for expander body

## Testing

```bash
cd backend
source venv/bin/activate
pytest tests/test_api.py -v
```

Manual demo (two terminals):

```bash
# Terminal 1
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Reviewer flow: open browser, upload scheme PDF, ask eligibility question, expand Sources panel.

## Out of scope

- Voice I/O, topic sidebars, suggestion chips
- Multi-document library, auth, rerank log UI

## Follow-ups

- Chat history persistence
- PDF preview alongside sources
- Deploy backend + frontend as single docker-compose
