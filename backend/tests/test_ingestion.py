from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion import ingest_pdf


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(path: Path, pages: list[str]) -> None:
    """Write a minimal multi-page PDF with extractable text (ASCII)."""
    page_count = len(pages)
    font_obj = 3 + page_count * 2
    objects: dict[int, str] = {}
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(page_count))
    objects[1] = "<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"
    objects[font_obj] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, text in enumerate(pages):
        page_id = 3 + i * 2
        content_id = page_id + 1
        wrapped = _escape_pdf_text(text)
        stream = f"BT /F1 12 Tf 50 750 Td ({wrapped}) Tj ET"
        objects[content_id] = (
            f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )

    body = bytearray()
    offsets = {0: 0}
    for obj_id in range(1, font_obj + 1):
        offsets[obj_id] = len(body)
    header = b"%PDF-1.4\n"
    payload = bytearray()
    for obj_id in range(1, font_obj + 1):
        offsets[obj_id] = len(header) + len(payload)
        payload.extend(f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n".encode("latin-1"))

    xref_start = len(header) + len(payload)
    count = font_obj + 1
    xref_lines = [f"xref\n0 {count}\n0000000000 65535 f \n"]
    for obj_id in range(1, font_obj + 1):
        xref_lines.append(f"{offsets[obj_id]:010d} 00000 n \n")
    trailer = (
        f"trailer\n<< /Size {count} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    path.write_bytes(header + bytes(payload) + "".join(xref_lines).encode("latin-1") + trailer.encode("ascii"))


def test_ingest_pdf_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError):
        ingest_pdf(missing)


def test_ingest_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("hello")
    with pytest.raises(ValueError, match="pdf"):
        ingest_pdf(other)


def test_page_numbers_source_and_chunk_index(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scheme-guide.pdf"
    _write_pdf(
        pdf_path,
        [
            "Eligibility: applicant must be a resident of the notified district.",
            "Application process: submit Form A at the tehsil office with ID proof.",
        ],
    )

    chunks = ingest_pdf(pdf_path)

    assert chunks, "expected at least one chunk"
    assert all(c.source == "scheme-guide.pdf" for c in chunks)
    pages = {c.page_number for c in chunks}
    assert 1 in pages
    assert 2 in pages
    assert all(c.page_number >= 1 for c in chunks)
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))
    page1_text = " ".join(c.text for c in chunks if c.page_number == 1)
    page2_text = " ".join(c.text for c in chunks if c.page_number == 2)
    assert "Eligibility" in page1_text
    assert "Application process" in page2_text


def test_empty_page_is_skipped(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with-blank.pdf"
    _write_pdf(
        pdf_path,
        [
            "Only page with content about PM-KISAN eligibility.",
            "",
        ],
    )

    chunks = ingest_pdf(pdf_path)

    assert chunks
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_overlap_on_long_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "long.pdf"
    sentence = "The scheme provides income support to eligible landholding farmer families. "
    _write_pdf(pdf_path, [sentence * 80])

    chunks = ingest_pdf(pdf_path)

    assert len(chunks) >= 2
    assert chunks[1].text[:80] in chunks[0].text
