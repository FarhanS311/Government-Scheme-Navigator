from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.document_store import UPLOAD_DIR, clear_index
from app.main import app
from app.rag_chain import RAGAnswer, SourceCitation
from tests.test_ingestion import _write_pdf

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_data_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = tmp_path / "index"
    upload_dir = tmp_path / "uploads"
    index_dir.mkdir()
    upload_dir.mkdir()
    monkeypatch.setattr("app.document_store.DEFAULT_INDEX_DIR", index_dir)
    monkeypatch.setattr("app.document_store.UPLOAD_DIR", upload_dir)
    clear_index(index_dir)


def test_status_not_ready(tmp_path: Path) -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json() == {
        "ready": False,
        "filename": None,
        "chunk_count": None,
    }


def test_upload_builds_index(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme.pdf"
    _write_pdf(
        pdf_path,
        ["Eligibility: resident farmer with valid land records."],
    )

    with pdf_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("scheme.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "scheme.pdf"
    assert payload["chunk_count"] > 0

    status = client.get("/api/status").json()
    assert status["ready"] is True
    assert status["filename"] == "scheme.pdf"
    assert status["chunk_count"] == payload["chunk_count"]


def test_upload_rejects_non_pdf() -> None:
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_ask_without_index_returns_409() -> None:
    response = client.post("/api/ask", json={"question": "Who is eligible?"})
    assert response.status_code == 409


def test_ask_returns_sources_with_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme.pdf"
    _write_pdf(
        pdf_path,
        ["Application process: submit Form A with identity proof."],
    )

    with pdf_path.open("rb") as handle:
        upload_response = client.post(
            "/api/upload",
            files={"file": ("scheme.pdf", handle, "application/pdf")},
        )
    assert upload_response.status_code == 200

    mock_answer = RAGAnswer(
        answer="Submit Form A with identity proof.",
        sources=[
            SourceCitation(id=0, page=1, snippet="Application process: submit Form A"),
        ],
        query_type="single_fact",
        sub_queries=["Who is eligible?"],
        contexts=["Application process: submit Form A with identity proof."],
    )

    with patch("app.main.ask_question", return_value=mock_answer):
        response = client.post("/api/ask", json={"question": "How do I apply?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == mock_answer.answer
    assert payload["query_type"] == "single_fact"
    assert payload["sub_queries"] == ["Who is eligible?"]
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["text"] == mock_answer.contexts[0]
    assert payload["sources"][0]["page"] == 1
