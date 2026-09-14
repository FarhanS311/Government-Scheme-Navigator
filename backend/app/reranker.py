from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.vector_store import ScoredChunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 20
DEFAULT_RERANK_LOG_PATH = Path("data/logs/rerank.jsonl")

_cross_encoder: CrossEncoder | None = None


@dataclass(frozen=True)
class RerankEntry:
    chunk_index: int
    page: int
    bi_encoder_score: float
    cross_encoder_score: float
    rank_before: int
    rank_after: int


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def rerank_chunks(
    question: str,
    scored: list[ScoredChunk],
    top_k: int = 4,
    cross_encoder: CrossEncoder | None = None,
) -> tuple[list[ScoredChunk], list[RerankEntry]]:
    if not scored:
        return [], []

    encoder = cross_encoder or _get_cross_encoder()
    ranked_before = sorted(scored, key=lambda item: item.score, reverse=True)
    rank_before_map = {
        item.chunk.chunk_index: index + 1 for index, item in enumerate(ranked_before)
    }

    pairs = [(question, item.chunk.text) for item in ranked_before]
    cross_scores = encoder.predict(pairs)
    scored_with_cross = list(zip(ranked_before, cross_scores, strict=True))
    scored_with_cross.sort(key=lambda item: float(item[1]), reverse=True)

    entries: list[RerankEntry] = []
    reranked: list[ScoredChunk] = []
    for rank_after, (item, cross_score) in enumerate(scored_with_cross, start=1):
        chunk = item.chunk
        entries.append(
            RerankEntry(
                chunk_index=chunk.chunk_index,
                page=chunk.page_number,
                bi_encoder_score=item.score,
                cross_encoder_score=float(cross_score),
                rank_before=rank_before_map[chunk.chunk_index],
                rank_after=rank_after,
            )
        )
        if rank_after <= top_k:
            reranked.append(ScoredChunk(chunk=chunk, score=float(cross_score)))

    return reranked, entries


def format_rerank_comparison(question: str, entries: list[RerankEntry]) -> str:
    before_lines = sorted(entries, key=lambda entry: entry.rank_before)
    after_lines = sorted(entries, key=lambda entry: entry.rank_after)

    lines = [f'Re-rank (query: "{question}")', "BEFORE                          AFTER"]
    width = 32
    for before, after in zip(before_lines, after_lines, strict=False):
        left = (
            f"#{before.rank_before} chunk={before.chunk_index} page={before.page} "
            f"score={before.bi_encoder_score:.4f}"
        )
        right = (
            f"#{after.rank_after} chunk={after.chunk_index} page={after.page} "
            f"score={after.cross_encoder_score:.4f}"
        )
        lines.append(f"{left:<{width}}{right}")
    return "\n".join(lines)


def log_rerank_to_jsonl(
    question: str,
    entries: list[RerankEntry],
    path: Path = DEFAULT_RERANK_LOG_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "entries": [
            {
                "chunk_index": entry.chunk_index,
                "page": entry.page,
                "bi_encoder_score": entry.bi_encoder_score,
                "cross_encoder_score": entry.cross_encoder_score,
                "rank_before": entry.rank_before,
                "rank_after": entry.rank_after,
            }
            for entry in entries
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
