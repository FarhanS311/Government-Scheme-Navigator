from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_llm
from app.evaluation import (
    DEFAULT_EVAL_LOG_PATH,
    DEFAULT_GOLDEN_SET_PATH,
    build_eval_record,
    evaluate_single,
    format_metrics_table,
    load_golden_set,
    log_eval_record,
)
from app.rag_chain import ask_question
from app.vector_store import DEFAULT_INDEX_DIR, load_index


def run_eval(
    golden_set_path: Path = DEFAULT_GOLDEN_SET_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    log_path: Path = DEFAULT_EVAL_LOG_PATH,
    k: int = 4,
) -> list:
    golden_items = load_golden_set(golden_set_path)
    index = load_index(index_dir)
    llm = get_llm()
    records = []

    for item in golden_items:
        result = ask_question(item["question"], index, llm=llm, k=k)
        metrics = evaluate_single(
            question=item["question"],
            answer=result.answer,
            contexts=result.contexts,
            ground_truth=item["ground_truth"],
            llm=llm,
        )
        record = build_eval_record(
            eval_id=item["id"],
            question=item["question"],
            answer=result.answer,
            ground_truth=item["ground_truth"],
            metrics=metrics,
        )
        log_eval_record(record, log_path)
        records.append(record)

    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run batch RAGAS evaluation on the golden set.")
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
        help=f"Path to golden eval set JSON (default: {DEFAULT_GOLDEN_SET_PATH})",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help=f"FAISS index directory (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_EVAL_LOG_PATH,
        help=f"JSONL output path (default: {DEFAULT_EVAL_LOG_PATH})",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=4,
        help="Number of chunks to send to the LLM after re-ranking (default: 4)",
    )
    args = parser.parse_args(argv)

    records = run_eval(
        golden_set_path=args.golden_set,
        index_dir=args.index_dir,
        log_path=args.log_path,
        k=args.k,
    )
    print(format_metrics_table(records))
    print(f"\nLogged {len(records)} evaluation(s) to {args.log_path}")


if __name__ == "__main__":
    main()
