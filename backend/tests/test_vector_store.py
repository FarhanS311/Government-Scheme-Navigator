from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion import Chunk
from app.vector_store import (
    INDEX_NAME,
    ScoredChunk,
    build_index,
    load_index,
)


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
        Chunk(
            text=(
                "Budget allocation for the scheme increased in the previous "
                "fiscal year according to the ministry annual report."
            ),
            source="scheme.pdf",
            page_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def indexed_dir(tmp_path: Path) -> Path:
    index_dir = tmp_path / "index"
    build_index(_sample_chunks(), index_dir)
    return index_dir


def test_build_index_persists_files(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(_sample_chunks(), index_dir)

    assert (index_dir / f"{INDEX_NAME}.faiss").exists()
    assert (index_dir / f"{INDEX_NAME}.pkl").exists()


def test_build_index_rejects_empty_chunks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_index([], tmp_path / "index")


def test_load_index_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="index directory not found"):
        load_index(tmp_path / "missing")


def test_load_index_missing_files_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="index files not found"):
        load_index(empty_dir)


def test_reload_returns_results(indexed_dir: Path) -> None:
    index = load_index(indexed_dir)
    results = index.query_index("eligibility criteria for farmers", k=2)

    assert len(results) == 2
    assert all(isinstance(item, ScoredChunk) for item in results)
    assert all(isinstance(item.chunk, Chunk) for item in results)


def test_query_index_scores_are_non_increasing(indexed_dir: Path) -> None:
    index = load_index(indexed_dir)
    results = index.query_index("application documents required", k=3)
    scores = [item.score for item in results]

    assert scores == sorted(scores, reverse=True)


def test_query_index_is_deterministic(indexed_dir: Path) -> None:
    index = load_index(indexed_dir)
    question = "Who is eligible for this scheme?"
    first = [(item.chunk.chunk_index, item.score) for item in index.query_index(question, k=2)]
    second = [(item.chunk.chunk_index, item.score) for item in index.query_index(question, k=2)]

    assert first == second


def test_query_index_discriminates_questions(indexed_dir: Path) -> None:
    index = load_index(indexed_dir)
    eligibility = index.query_index("eligibility criteria", k=1)[0].chunk.chunk_index
    application = index.query_index("application process", k=1)[0].chunk.chunk_index

    assert eligibility != application
    assert eligibility == 0
    assert application == 1
