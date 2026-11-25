"""Pydantic request and response models for the FastAPI HTTP layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceOut(BaseModel):
    """A retrieved chunk exposed to clients for citation UI."""

    id: int
    page: int
    snippet: str
    text: str


class ChatTurn(BaseModel):
    """One prior user or assistant message in a conversational ask request."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    """Body for ``/api/ask`` and ``/api/ask/stream``."""

    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)


class AskResponse(BaseModel):
    """JSON answer payload including routing metadata and source chunks."""

    answer: str
    query_type: str
    sub_queries: list[str]
    sources: list[SourceOut]


class UploadResponse(BaseModel):
    """Outcome of indexing an uploaded scheme PDF."""

    filename: str
    chunk_count: int
    message: str


class StatusResponse(BaseModel):
    """Whether a document is loaded and basic index statistics."""

    ready: bool
    filename: str | None = None
    chunk_count: int | None = None