from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from app.query_router import RouteResult, _normalize_route_result, route_query


def test_route_multi_part_parses_sub_queries() -> None:
    expected = RouteResult(
        query_type="multi_part",
        sub_queries=[
            "What is the eligibility criteria?",
            "How do you apply for the scheme?",
        ],
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: expected)

    result = route_query(
        "What is the eligibility criteria and how do you apply?",
        mock_llm,
    )

    assert result.query_type == "multi_part"
    assert len(result.sub_queries) == 2


def test_route_single_fact_returns_original() -> None:
    question = "Who is eligible for this scheme?"
    expected = RouteResult(query_type="single_fact", sub_queries=[question])
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _: expected)

    result = route_query(question, mock_llm)

    assert result.query_type == "single_fact"
    assert result.sub_queries == [question]


def test_route_fallback_on_router_failure() -> None:
    question = "Who is eligible?"
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = RuntimeError("structured output failed")

    result = route_query(question, mock_llm)

    assert result.query_type == "single_fact"
    assert result.sub_queries == [question]


def test_normalize_route_result_rejects_empty_sub_queries() -> None:
    result = _normalize_route_result(
        "Who is eligible?",
        RouteResult(query_type="multi_part", sub_queries=["", "  "]),
    )

    assert result.query_type == "single_fact"
    assert result.sub_queries == ["Who is eligible?"]
