from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion import ingest_pdf
from app.vector_store import DEFAULT_INDEX_DIR, VectorIndex, build_index, load_index

UPLOAD_DIR = Path("data/uploads")
META_FILENAME = "document_meta.json"

_index_cache: VectorIndex | None = None


def _resolve_index_dir(index_dir: Path | None) -> Path:
    return index_dir if index_dir is not None else DEFAULT_INDEX_DIR


def _resolve_upload_dir(upload_dir: Path | None) -> Path:
    return upload_dir if upload_dir is not None else UPLOAD_DIR


def _meta_path(index_dir: Path) -> Path:
    return index_dir / META_FILENAME


def _read_meta(index_dir: Path) -> dict[str, object] | None:
    path = _meta_path(index_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_meta(filename: str, chunk_count: int, index_dir: Path) -> None:
    payload = {
        "filename": filename,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    _meta_path(index_dir).write_text(
        json.dumps(payload, ensure_ascii=True),
        encoding="utf-8",
    )


def get_status(index_dir: Path | None = None) -> dict[str, object]:
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
    global _index_cache
    _index_cache = None
    resolved = _resolve_index_dir(index_dir)
    if resolved.exists():
        shutil.rmtree(resolved)


def get_index(index_dir: Path | None = None) -> VectorIndex:
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
    global _index_cache

    resolved_index_dir = _resolve_index_dir(index_dir)
    resolved_upload_dir = _resolve_upload_dir(upload_dir)

    if not filename.lower().endswith(".pdf"):
        raise ValueError("only PDF files are supported")

    resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    clear_index(resolved_index_dir)

    pdf_path = resolved_upload_dir / filename
    pdf_path.write_bytes(file_bytes)

    chunks = ingest_pdf(pdf_path)
    if not chunks:
        raise ValueError("PDF produced no text chunks")

    build_index(chunks, resolved_index_dir)
    _write_meta(filename, len(chunks), resolved_index_dir)
    _index_cache = load_index(resolved_index_dir)

    return {
        "filename": filename,
        "chunk_count": len(chunks),
        "message": f"Indexed {len(chunks)} chunk(s) from {filename}",
    }
