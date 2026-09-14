from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import get_llm
from app.ingestion import Chunk
from app.query_router import RouteResult
from app.reranker import RerankEntry
from app.rag_chain import (
    NOT_IN_DOCUMENT_PHRASE,
    _format_context,
    _merge_scored_chunks,
    _retrieve_multi,
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


def _passthrough_rerank(
    question: str, scored: list[ScoredChunk], top_k: int = 4, cross_encoder: object = None
) -> tuple[list[ScoredChunk], list[RerankEntry]]:
    ordered = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
    entries = [
        RerankEntry(
            chunk_index=item.chunk.chunk_index,
            page=item.chunk.page_number,
            bi_encoder_score=item.score,
            cross_encoder_score=item.score,
            rank_before=index + 1,
            rank_after=index + 1,
        )
        for index, item in enumerate(ordered)
    ]
    return ordered, entries


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


@patch("app.rag_chain.rerank_chunks", side_effect=_passthrough_rerank)
@patch("app.rag_chain.route_query")
def test_ask_question_in_scope(
    mock_route: object, mock_rerank: object, indexed_dir: Path
) -> None:
    mock_route.return_value = RouteResult(
        query_type="single_fact",
        sub_queries=["Who is eligible for this scheme?"],
    )
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
    assert result.query_type == "single_fact"


@patch("app.rag_chain.rerank_chunks", side_effect=_passthrough_rerank)
@patch("app.rag_chain.route_query")
def test_ask_question_out_of_scope(
    mock_route: object, mock_rerank: object, indexed_dir: Path
) -> None:
    mock_route.return_value = RouteResult(
        query_type="single_fact",
        sub_queries=["What is the capital of France?"],
    )
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


def test_merge_scored_chunks_dedupes_by_chunk_index() -> None:
    chunk_a = _sample_chunks()[0]
    chunk_b = _sample_chunks()[1]
    merged = _merge_scored_chunks(
        [
            ScoredChunk(chunk=chunk_a, score=0.5),
            ScoredChunk(chunk=chunk_a, score=0.9),
            ScoredChunk(chunk=chunk_b, score=0.7),
        ],
        k=2,
    )

    assert len(merged) == 2
    assert merged[0].chunk.chunk_index == 0
    assert merged[0].score == 0.9
    assert merged[1].chunk.chunk_index == 1


@patch("app.rag_chain.rerank_chunks", side_effect=_passthrough_rerank)
@patch("app.rag_chain.route_query")
def test_multi_part_retrieves_both_chunk_groups(
    mock_route: object, mock_rerank: object, indexed_dir: Path
) -> None:
    mock_route.return_value = RouteResult(
        query_type="multi_part",
        sub_queries=["eligibility criteria", "application process"],
    )
    fake_llm = FakeListChatModel(
        responses=[
            "Eligibility is on page 1. Apply using Form A at the tehsil office on page 2."
        ]
    )
    index = load_index(indexed_dir)
    result = ask_question(
        "What is eligibility and how do you apply?",
        index,
        llm=fake_llm,
        k=2,
    )

    assert result.query_type == "multi_part"
    assert len(result.sub_queries) >= 2
    source_ids = {source.id for source in result.sources}
    assert 0 in source_ids
    assert 1 in source_ids


def test_retrieve_multi_merges_sub_queries(indexed_dir: Path) -> None:
    index = load_index(indexed_dir)
    scored = _retrieve_multi(
        index,
        ["eligibility criteria", "application process"],
        k=2,
    )

    chunk_indexes = {item.chunk.chunk_index for item in scored}
    assert 0 in chunk_indexes
    assert 1 in chunk_indexes


@patch("app.rag_chain.rerank_chunks", side_effect=_passthrough_rerank)
@patch("app.rag_chain.route_query")
def test_ask_question_includes_rerank_log(
    mock_route: object, mock_rerank: object, indexed_dir: Path
) -> None:
    mock_route.return_value = RouteResult(
        query_type="single_fact",
        sub_queries=["Who is eligible?"],
    )
    fake_llm = FakeListChatModel(responses=["Eligible farmers on page 1."])
    index = load_index(indexed_dir)
    result = ask_question("Who is eligible?", index, llm=fake_llm, k=2)

    assert result.rerank_log
    mock_rerank.assert_called_once()


@patch("app.rag_chain.rerank_chunks", side_effect=_passthrough_rerank)
@patch("app.rag_chain.route_query")
def test_rag_answer_includes_contexts(
    mock_route: object, mock_rerank: object, indexed_dir: Path
) -> None:
    mock_route.return_value = RouteResult(
        query_type="single_fact",
        sub_queries=["Who is eligible?"],
    )
    fake_llm = FakeListChatModel(responses=["Eligible farmers on page 1."])
    index = load_index(indexed_dir)
    result = ask_question("Who is eligible?", index, llm=fake_llm, k=2)

    assert result.contexts
    assert any("Eligibility criteria" in context for context in result.contexts)


def test_missing_api_key_raises() -> None:
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            get_llm()
