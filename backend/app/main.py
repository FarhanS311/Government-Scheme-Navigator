"""FastAPI HTTP API for PDF upload, status, and grounded Q&A (JSON and SSE)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import get_llm
from app.document_store import get_index, get_status, ingest_upload
from app.rag_chain import (
    NOT_IN_DOCUMENT_PHRASE,
    PreparedRAG,
    ask_question,
    prepare_rag_state,
    stream_answer_tokens,
)
from app.schemas import AskRequest, AskResponse, SourceOut, StatusResponse, UploadResponse

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Government Scheme Navigator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sources_from_prepared(prepared: PreparedRAG) -> list[SourceOut]:
    """Map internal citations and chunk texts to API source objects."""
    sources: list[SourceOut] = []
    for citation, text in zip(prepared.sources, prepared.contexts, strict=False):
        sources.append(
            SourceOut(
                id=citation.id,
                page=citation.page,
                snippet=citation.snippet,
                text=text,
            )
        )
    if len(prepared.sources) > len(prepared.contexts):
        for citation in prepared.sources[len(prepared.contexts) :]:
            sources.append(
                SourceOut(
                    id=citation.id,
                    page=citation.page,
                    snippet=citation.snippet,
                    text=citation.snippet,
                )
            )
    return sources


def _require_index() -> None:
    """Raise HTTP 409 when no PDF has been indexed yet."""
    status_data = get_status()
    if not status_data["ready"]:
        raise HTTPException(
            status_code=409,
            detail="no document indexed; upload a PDF first",
        )


def _sse(event: str, data: object) -> str:
    """Format one Server-Sent Events message block."""
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    """Return whether a scheme PDF is indexed and how many chunks it produced."""
    data = get_status()
    return StatusResponse(**data)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF upload, validate size, and index it in a worker thread."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload exceeds 20 MB limit")

    try:
        result = await asyncio.to_thread(ingest_upload, file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return UploadResponse(**result)


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    """Run the full RAG pipeline and return a JSON answer with sources."""
    _require_index()

    index = get_index()
    result = ask_question(
        body.question.strip(),
        index,
        k=body.k,
        history=body.history,
    )

    sources: list[SourceOut] = []
    for citation, text in zip(result.sources, result.contexts, strict=False):
        sources.append(
            SourceOut(
                id=citation.id,
                page=citation.page,
                snippet=citation.snippet,
                text=text,
            )
        )
    if len(result.sources) > len(result.contexts):
        for citation in result.sources[len(result.contexts) :]:
            sources.append(
                SourceOut(
                    id=citation.id,
                    page=citation.page,
                    snippet=citation.snippet,
                    text=citation.snippet,
                )
            )

    return AskResponse(
        answer=result.answer,
        query_type=result.query_type,
        sub_queries=result.sub_queries,
        sources=sources,
    )


@app.post("/api/ask/stream")
async def ask_stream(body: AskRequest) -> StreamingResponse:
    """Stream retrieval metadata, answer tokens, sources, and a done event over SSE."""
    _require_index()

    index = get_index()
    llm = get_llm()
    prepared = await asyncio.to_thread(
        prepare_rag_state,
        body.question.strip(),
        index,
        llm,
        body.k,
        body.history,
    )

    async def event_generator() -> AsyncIterator[str]:
        """Yield SSE frames for meta, tokens, sources, and completion."""
        yield _sse(
            "meta",
            {"query_type": prepared.query_type, "sub_queries": prepared.sub_queries},
        )

        if prepared.abstain:
            yield _sse("token", {"text": NOT_IN_DOCUMENT_PHRASE})
        else:
            async for token in stream_answer_tokens(prepared, llm):
                if token:
                    yield _sse("token", {"text": token})

        sources = [source.model_dump() for source in _sources_from_prepared(prepared)]
        yield _sse("sources", sources)
        yield _sse("done", {})

    return StreamingResponse(event_generator(), media_type="text/event-stream")