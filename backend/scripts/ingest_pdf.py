from __future__ import annotations

import argparse
from pathlib import Path

from app.ingestion import ingest_pdf


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load a PDF and print text chunks with page metadata.")
    parser.add_argument("pdf_path", type=Path, help="Path to a PDF file")
    args = parser.parse_args(argv)

    chunks = ingest_pdf(args.pdf_path)
    print(f"Loaded {len(chunks)} chunk(s) from {args.pdf_path.name}")
    print("=" * 60)
    for chunk in chunks:
        print(
            f"[chunk_index={chunk.chunk_index}] source={chunk.source} "
            f"page={chunk.page_number} chars={len(chunk.text)}"
        )
        print(chunk.text)
        print("-" * 60)


if __name__ == "__main__":
    main()
