from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.rag_chain import ask_question
from app.vector_store import DEFAULT_INDEX_DIR, load_index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question using the LCEL RAG chain over the persisted index."
    )
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"Directory containing the FAISS index (default: {DEFAULT_INDEX_DIR})",
    )
    args = parser.parse_args(argv)

    index = load_index(args.index_dir)
    result = ask_question(args.question, index, k=args.k)

    print(f"Question: {args.question}")
    print("=" * 60)
    print("Answer:")
    print(result.answer)
    print("=" * 60)
    print("Sources:")
    for source in result.sources:
        print(
            json.dumps(
                {"id": source.id, "page": source.page, "snippet": source.snippet},
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
