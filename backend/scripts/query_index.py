from __future__ import annotations

import argparse
from pathlib import Path

from app.vector_store import DEFAULT_INDEX_DIR, load_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Query a persisted FAISS index and print top-k chunks."
    )
    parser.add_argument("question", type=str, help="Question to search for")
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of chunks to return (default: 4)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing the FAISS index (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args(argv)

    index = load_index(args.index_dir)
    results = index.query_index(args.question, k=args.k)

    print(f"Question: {args.question}")
    print(f"Top {len(results)} result(s) from {args.index_dir}")
    print("=" * 60)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = chunk.text[:120].replace("\n", " ")
        print(
            f"[rank={rank}] score={result.score:.4f} chunk_index={chunk.chunk_index} "
            f"source={chunk.source} page={chunk.page_number}"
        )
        print(preview)
        print("-" * 60)


if __name__ == "__main__":
    main()
