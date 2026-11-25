"""Single-document upload handling, on-disk metadata, and in-memory index cache."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion import ingest_pdf
from app.vector_store import DEFAULT_INDEX_DIR, VectorIndex, build_index, load_index

UPLOAD_DIR = Path("data/uploads")
META_FILENAME = "document_meta.json"
STORED_PDF_FILENAME = "document.pdf"

_index_cache: VectorIndex | None = None
_upload_lock = threading.Lock()


def _resolve_index_dir(index_dir: Path | None) -> Path:
    """Use the default index directory when ``index_dir`` is omitted."""
    return index_dir if index_dir is not None else DEFAULT_INDEX_DIR


def _resolve_upload_dir(upload_dir: Path | None) -> Path:
    """Use the default upload directory when ``upload_dir`` is omitted."""
    return upload_dir if upload_dir is not None else UPLOAD_DIR


def _meta_path(index_dir: Path) -> Path:
    """Return the JSON metadata file path for an index directory."""
    return index_dir / META_FILENAME


def _read_meta(index_dir: Path) -> dict[str, object] | None:
    """Load index metadata or return None if no document has been indexed."""
    path = _meta_path(index_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_meta(filename: str, chunk_count: int, index_dir: Path) -> None:
    """Persist display filename, chunk count, and upload timestamp."""
    payload = {
        "filename": filename,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _meta_path(index_dir).write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def sanitize_display_filename(filename: str) -> str:
    """Keep only a safe PDF basename for API responses and metadata."""
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("invalid filename")
    if not name.lower().endswith(".pdf"):
        raise ValueError("only PDF files are supported")
    if "/" in name or "\\" in name:
        raise ValueError("invalid filename")
    return name


def get_status(index_dir: Path | None = None) -> dict[str, object]:
    """Report whether an index exists and basic document statistics."""
    resolved = _resolve_index_dir(index_dir)
    meta = _read_meta(resolved)
    if meta is None:
        return {"ready": False, "filename": None, "chunk_count": None}
    return {
        "ready": True,
        "filename": meta["filename"],
        "chunk_count": meta["chunk_count"],
    }


def clear_index(index_dir: Path | None = None) -> None:
    """Remove persisted FAISS files and drop the in-memory index cache."""
    global _index_cache
    _index_cache = None
    resolved = _resolve_index_dir(index_dir)
    if resolved.exists():
        shutil.rmtree(resolved)


def get_index(index_dir: Path | None = None) -> VectorIndex:
    """Return a cached ``VectorIndex`` or load it from disk."""
    global _index_cache
    resolved = _resolve_index_dir(index_dir)
    if _index_cache is None:
        _index_cache = load_index(resolved)
    return _index_cache


def ingest_upload(
    file_bytes: bytes,
    filename: str,
    index_dir: Path | None = None,
    upload_dir: Path | None = None,
) -> dict[str, object]:
    """Replace the current document: write PDF, chunk, build FAISS, update meta."""
    global _index_cache

    display_name = sanitize_display_filename(filename)
    resolved_index_dir = _resolve_index_dir(index_dir)
    resolved_upload_dir = _resolve_upload_dir(upload_dir)

    with _upload_lock:
        resolved_upload_dir.mkdir(parents=True, exist_ok=True)
        clear_index(resolved_index_dir)

        pdf_path = resolved_upload_dir / STORED_PDF_FILENAME
        pdf_path.write_bytes(file_bytes)

        chunks = ingest_pdf(pdf_path)
        if not chunks:
            raise ValueError("PDF produced no text chunks")

        build_index(chunks, resolved_index_dir)
        _write_meta(display_name, len(chunks), resolved_index_dir)
        _index_cache = load_index(resolved_index_dir)

        return {
            "filename": display_name,
            "chunk_count": len(chunks),
            "message": f"Indexed {len(chunks)} chunk(s) from {display_name}",
        }