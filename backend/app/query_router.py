from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

ROUTER_PROMPT = """Classify the user question for a government scheme document RAG system.

Return JSON with:
- query_type: one of single_fact, multi_part, summarization
- sub_queries: list of search queries

Rules:
- single_fact: one specific fact; sub_queries = [original question]
- multi_part: two or more distinct questions joined (e.g. "X and Y"); split into separate sub_queries
- summarization: broad overview request; sub_queries = [original question]
- sub_queries must be non-empty strings
- for multi_part, return at least 2 sub_queries

Question: {question}"""


class RouteResult(BaseModel):
    query_type: Literal["single_fact", "multi_part", "summarization"]
    sub_queries: list[str] = Field(min_length=1)


def _normalize_route_result(question: str, result: RouteResult) -> RouteResult:
    sub_queries = [part.strip() for part in result.sub_queries if part.strip()]
    if not sub_queries:
        return RouteResult(query_type="single_fact", sub_queries=[question])
    if result.query_type == "multi_part" and len(sub_queries) < 2:
        return RouteResult(query_type="single_fact", sub_queries=[question])
    return RouteResult(query_type=result.query_type, sub_queries=sub_queries)


def route_query(question: str, llm: BaseChatModel) -> RouteResult:
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    try:
        router = llm.with_structured_output(RouteResult)
        result = (prompt | router).invoke({"question": question})
        if isinstance(result, dict):
            result = RouteResult.model_validate(result)
        return _normalize_route_result(question, result)
    except Exception:
        return RouteResult(query_type="single_fact", sub_queries=[question])
