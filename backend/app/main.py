from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.document_store import get_index, get_status, ingest_upload
from app.rag_chain import ask_question
from app.schemas import AskRequest, AskResponse, SourceOut, StatusResponse, UploadResponse

app = FastAPI(title="Government Scheme Navigator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status", response_model=StatusResponse)
def status() -> StatusResponse:
    data = get_status()
    return StatusResponse(**data)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    try:
        result = ingest_upload(file_bytes, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return UploadResponse(**result)


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    status_data = get_status()
    if not status_data["ready"]:
        raise HTTPException(
            status_code=409,
            detail="no document indexed; upload a PDF first",
        )

    index = get_index()
    result = ask_question(body.question.strip(), index, k=body.k)

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
