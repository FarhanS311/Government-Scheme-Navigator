from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import document_store
from app.document_store import STORED_PDF_FILENAME, clear_index
from app.main import app
from app.rag_chain import PreparedRAG, RAGAnswer, SourceCitation
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


def test_upload_sanitizes_filename_and_stores_document_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme.pdf"
    _write_pdf(
        pdf_path,
        ["Eligibility: resident farmer with valid land records."],
    )

    with pdf_path.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": ("../secret.pdf", handle, "application/pdf")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "secret.pdf"
    assert (document_store.UPLOAD_DIR / STORED_PDF_FILENAME).is_file()


def test_upload_rejects_oversized_body(tmp_path: Path) -> None:
    pdf_path = tmp_path / "big.pdf"
    _write_pdf(pdf_path, ["short text"])
    oversized = pdf_path.read_bytes() + (b"0" * (21 * 1024 * 1024))

    response = client.post(
        "/api/upload",
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413


def test_ask_stream_emits_token_then_sources(tmp_path: Path) -> None:
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
        sub_queries=["How do I apply?"],
        contexts=["Application process: submit Form A with identity proof."],
    )
    prepared = PreparedRAG(
        question="How do I apply?",
        search_question="How do I apply?",
        history=[],
        query_type="single_fact",
        sub_queries=["How do I apply?"],
        reranked=[],
        rerank_log=[],
        abstain=False,
        context="ctx",
        contexts=mock_answer.contexts,
        sources=mock_answer.sources,
    )

    async def fake_stream(_prepared: object, _llm: object):
        yield "Submit "
        yield "Form A."

    with (
        patch("app.main.prepare_rag_state", return_value=prepared),
        patch("app.main.stream_answer_tokens", side_effect=fake_stream),
        patch("app.main.get_llm"),
    ):
        with client.stream(
            "POST",
            "/api/ask/stream",
            json={"question": "How do I apply?"},
        ) as response:
            assert response.status_code == 200
            events: list[tuple[str, object]] = []
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event_type = "message"
                    data_line = ""
                    for line in block.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_line = line[5:].strip()
                    if data_line:
                        events.append((event_type, json.loads(data_line)))

    event_types = [event for event, _ in events]
    assert "meta" in event_types
    token_idx = event_types.index("token")
    sources_idx = event_types.index("sources")
    assert token_idx < sources_idx
    token_text = "".join(
        payload["text"] for kind, payload in events if kind == "token"
    )
    assert "Form A" in token_text
    sources_payload = next(payload for kind, payload in events if kind == "sources")
    assert sources_payload
    assert event_types[-1] == "done"
