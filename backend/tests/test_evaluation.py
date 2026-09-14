from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from app.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    EvalRecord,
    build_eval_record,
    format_metrics_table,
    load_golden_set,
    log_eval_record,
)
from scripts import run_eval as run_eval_script


def test_load_golden_set_has_minimum_rows() -> None:
    items = load_golden_set(DEFAULT_GOLDEN_SET_PATH)

    assert len(items) >= 8
    for item in items:
        assert {"id", "question", "ground_truth"} <= set(item.keys())


def test_log_eval_record_writes_jsonl(tmp_path: Path) -> None:
    record = EvalRecord(
        eval_id="eligibility-01",
        timestamp="2026-09-14T08:00:00+00:00",
        question="Who is eligible?",
        answer_preview="Resident farmers with Aadhaar.",
        ground_truth="Resident farmers with landholding records and Aadhaar.",
        faithfulness=0.91,
        answer_relevancy=0.87,
        context_precision=0.8,
    )
    log_path = tmp_path / "ragas.jsonl"
    log_eval_record(record, log_path)

    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["question"] == "Who is eligible?"
    assert payload["metrics"]["faithfulness"] == 0.91


def test_format_metrics_table_includes_mean_row() -> None:
    records = [
        build_eval_record(
            "eligibility-01",
            "Who is eligible?",
            "Farmers with Aadhaar.",
            "Resident farmers with Aadhaar.",
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
            },
        ),
        build_eval_record(
            "application-01",
            "How do I apply?",
            "Submit Form A.",
            "Submit Form A at the tehsil office.",
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": 0.6,
            },
        ),
    ]
    table = format_metrics_table(records)

    assert "faithfulness" in table
    assert "MEAN" in table
    assert "eligibility-01" in table


@patch("scripts.run_eval.run_eval")
def test_run_eval_mocked(mock_run_eval: object) -> None:
    mock_run_eval.return_value = [
        build_eval_record(
            "eligibility-01",
            "Who is eligible?",
            "Farmers with Aadhaar.",
            "Resident farmers with Aadhaar.",
            {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
            },
        )
    ]

    buffer = StringIO()
    with patch("sys.stdout", buffer):
        run_eval_script.main([])

    output = buffer.getvalue()
    assert "faithfulness" in output
    assert "Logged 1 evaluation(s)" in output
