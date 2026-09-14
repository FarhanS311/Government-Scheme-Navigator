from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import get_llm
from app.ingestion import Chunk
from app.rag_chain import (
    NOT_IN_DOCUMENT_PHRASE,
    _format_context,
    _to_citations,
    ask_question,
)
from app.vector_store import ScoredChunk, build_index, load_index


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text=(
                "Eligibility criteria: the applicant must be a resident farmer "
                "with landholding records and a valid Aadhaar card."
            ),
            source="scheme.pdf",
            page_number=1,
            chunk_index=0,
        ),
        Chunk(
            text=(
                "Application process: submit Form A at the tehsil office with "
                "identity proof, bank passbook, and land ownership documents."
            ),
            source="scheme.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]


def _scored_chunks() -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk, score=0.9 - index * 0.1) for index, chunk in enumerate(_sample_chunks())]


@pytest.fixture
def indexed_dir(tmp_path: Path) -> Path:
    index_dir = tmp_path / "index"
    build_index(_sample_chunks(), index_dir)
    return index_dir


def test_format_context_includes_page_and_chunk_index() -> None:
    context = _format_context(_scored_chunks())

    assert "[Chunk 0 | page 1]" in context
    assert "[Chunk 1 | page 2]" in context
    assert "Eligibility criteria" in context
    assert "Application process" in context


def test_to_citations_maps_fields() -> None:
    long_text = "x" * 250
    scored = [
        ScoredChunk(
            chunk=Chunk(
                text=long_text,
                source="scheme.pdf",
                page_number=4,
                chunk_index=7,
            ),
            score=0.5,
        )
    ]

    citations = _to_citations(scored, snippet_len=200)

    assert len(citations) == 1
    assert citations[0].id == 7
    assert citations[0].page == 4
    assert citations[0].snippet == "x" * 200


def test_ask_question_in_scope(indexed_dir: Path) -> None:
    fake_llm = FakeListChatModel(
        responses=[
            "Applicants must be resident farmers with landholding records and Aadhaar (page 1)."
        ]
    )
    index = load_index(indexed_dir)
    result = ask_question(
        "Who is eligible for this scheme?",
        index,
        llm=fake_llm,
        k=2,
    )

    assert "resident farmer" in result.answer.lower()
    assert result.sources
    assert any(source.page == 1 for source in result.sources)


def test_ask_question_out_of_scope(indexed_dir: Path) -> None:
    fake_llm = FakeListChatModel(responses=[NOT_IN_DOCUMENT_PHRASE])
    index = load_index(indexed_dir)
    result = ask_question(
        "What is the capital of France?",
        index,
        llm=fake_llm,
        k=2,
    )

    assert result.answer == NOT_IN_DOCUMENT_PHRASE
    assert result.sources


def test_missing_api_key_raises() -> None:
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            get_llm()
