from __future__ import annotations

from pathlib import Path
import pytest

from app.ingestion import Chunk
from app.reranker import (
    format_rerank_comparison,
    log_rerank_to_jsonl,
    rerank_chunks,
)
from app.vector_store import ScoredChunk


def _rerank_corpus() -> list[Chunk]:
    return [
        Chunk(
            text=(
                "Eligibility criteria: required documents include Aadhaar card and "
                "landholding records for resident farmers."
            ),
            source="scheme.pdf",
            page_number=1,
            chunk_index=0,
        ),
        Chunk(
            text=(
                "Application process: submit Form A at the tehsil office with identity "
                "proof, bank passbook, and land ownership documents."
            ),
            source="scheme.pdf",
            page_number=2,
            chunk_index=1,
        ),
    ]


def _scored_corpus() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=_rerank_corpus()[0], score=0.41),
        ScoredChunk(chunk=_rerank_corpus()[1], score=0.38),
    ]


class FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _, text in pairs:
            if "Application process" in text:
                scores.append(9.0)
            else:
                scores.append(1.0)
        return scores


def test_rerank_chunks_assigns_ranks() -> None:
    reranked, entries = rerank_chunks(
        "What documents are needed for the application process?",
        _scored_corpus(),
        top_k=2,
        cross_encoder=FakeCrossEncoder(),
    )

    assert len(reranked) == 2
    assert len(entries) == 2
    by_chunk = {entry.chunk_index: entry for entry in entries}
    assert by_chunk[0].rank_before == 1
    assert by_chunk[1].rank_before == 2
    assert reranked[0].chunk.chunk_index == 1


def test_rerank_changes_top_result() -> None:
    _, entries = rerank_chunks(
        "What documents are needed for the application process?",
        _scored_corpus(),
        top_k=2,
        cross_encoder=FakeCrossEncoder(),
    )

    before_top = min(entries, key=lambda entry: entry.rank_before)
    after_top = min(entries, key=lambda entry: entry.rank_after)
    assert before_top.chunk_index == 0
    assert after_top.chunk_index == 1


def test_format_rerank_comparison_includes_before_and_after() -> None:
    _, entries = rerank_chunks(
        "application documents",
        _scored_corpus(),
        top_k=2,
        cross_encoder=FakeCrossEncoder(),
    )
    text = format_rerank_comparison("application documents", entries)

    assert "BEFORE" in text
    assert "AFTER" in text
    assert "chunk=0" in text
    assert "chunk=1" in text


def test_log_rerank_to_jsonl_writes_record(tmp_path: Path) -> None:
    _, entries = rerank_chunks(
        "application documents",
        _scored_corpus(),
        top_k=2,
        cross_encoder=FakeCrossEncoder(),
    )
    log_path = tmp_path / "rerank.jsonl"
    log_rerank_to_jsonl("application documents", entries, log_path)

    content = log_path.read_text(encoding="utf-8")
    assert "application documents" in content
    assert "rank_before" in content


def test_rerank_with_real_cross_encoder_changes_top_result() -> None:
    question = "What documents are needed for the application process?"
    before = _scored_corpus()
    reranked, entries = rerank_chunks(question, before, top_k=2)

    top_before = min(entries, key=lambda entry: entry.rank_before)
    top_after = min(entries, key=lambda entry: entry.rank_after)
    assert top_before.chunk_index == 0
    assert top_after.chunk_index == 1
    assert reranked[0].chunk.chunk_index == 1
