from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion import ingest_pdf
from app.vector_store import DEFAULT_INDEX_DIR, build_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF and build a persisted FAISS index."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF file")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to persist the FAISS index (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args(argv)

    chunks = ingest_pdf(args.pdf_path)
    build_index(chunks, args.index_dir)
    print(f"Built index with {len(chunks)} chunk(s) at {args.index_dir}")


if __name__ == "__main__":
    main()
