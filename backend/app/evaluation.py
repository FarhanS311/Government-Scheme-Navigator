from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.vector_store import EMBEDDING_MODEL

DEFAULT_GOLDEN_SET_PATH = Path("data/eval/golden_set.json")
DEFAULT_EVAL_LOG_PATH = Path("data/logs/ragas.jsonl")
ANSWER_PREVIEW_LENGTH = 200
REQUIRED_GOLDEN_KEYS = {"id", "question", "ground_truth"}
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")


@dataclass(frozen=True)
class EvalRecord:
    eval_id: str
    timestamp: str
    question: str
    answer_preview: str
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> list[dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("golden set must be a JSON array")

    for item in items:
        missing = REQUIRED_GOLDEN_KEYS - set(item)
        if missing:
            raise ValueError(f"golden set item missing keys: {sorted(missing)}")

    return items


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    llm: BaseChatModel,
) -> dict[str, float]:
    wrapped_llm = LangchainLLMWrapper(llm)
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=wrapped_llm,
        embeddings=embeddings,
    )
    scores: dict[str, float] = {}
    result_frame = result.to_pandas()
    for metric_name in METRIC_NAMES:
        value = result_frame[metric_name].iloc[0]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if hasattr(value, "item"):
            scores[metric_name] = float(value.item())
        else:
            scores[metric_name] = float(value)
    return scores


def log_eval_record(record: EvalRecord, path: Path = DEFAULT_EVAL_LOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": record.timestamp,
        "question": record.question,
        "answer_preview": record.answer_preview,
        "ground_truth": record.ground_truth,
        "metrics": {
            "faithfulness": record.faithfulness,
            "answer_relevancy": record.answer_relevancy,
            "context_precision": record.context_precision,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def build_eval_record(
    eval_id: str,
    question: str,
    answer: str,
    ground_truth: str,
    metrics: dict[str, float],
) -> EvalRecord:
    return EvalRecord(
        eval_id=eval_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        question=question,
        answer_preview=answer[:ANSWER_PREVIEW_LENGTH],
        ground_truth=ground_truth,
        faithfulness=metrics.get("faithfulness"),
        answer_relevancy=metrics.get("answer_relevancy"),
        context_precision=metrics.get("context_precision"),
    )


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_metrics_table(records: list[EvalRecord]) -> str:
    header = (
        f"| {'id':<16} | {'faithfulness':<12} | "
        f"{'answer_relevancy':<16} | {'context_precision':<17} |"
    )
    separator = (
        f"|{'-' * 18}|{'-' * 14}|{'-' * 18}|{'-' * 19}|"
    )
    lines = [header, separator]
    for record in records:
        lines.append(
            f"| {record.eval_id:<16} | {_format_score(record.faithfulness):<12} | "
            f"{_format_score(record.answer_relevancy):<16} | "
            f"{_format_score(record.context_precision):<17} |"
        )

    means: dict[str, float | None] = {}
    for metric_name in METRIC_NAMES:
        values = [
            getattr(record, metric_name)
            for record in records
            if getattr(record, metric_name) is not None
        ]
        means[metric_name] = mean(values) if values else None

    lines.append(
        f"| {'MEAN':<16} | {_format_score(means['faithfulness']):<12} | "
        f"{_format_score(means['answer_relevancy']):<16} | "
        f"{_format_score(means['context_precision']):<17} |"
    )
    return "\n".join(lines)
