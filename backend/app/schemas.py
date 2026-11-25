from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SourceOut(BaseModel):
    id: int
    page: int
    snippet: str
    text: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=4, ge=1, le=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=6)


class AskResponse(BaseModel):
    answer: str
    query_type: str
    sub_queries: list[str]
    sources: list[SourceOut]


class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
    message: str


class StatusResponse(BaseModel):
    ready: bool
    filename: str | None = None
    chunk_count: int | None = None