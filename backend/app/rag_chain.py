from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.config import get_llm
from app.query_router import route_query
from app.reranker import RERANK_TOP_N, RerankEntry, rerank_chunks
from app.vector_store import ScoredChunk, VectorIndex

NOT_IN_DOCUMENT_PHRASE = "The answer is not in this document."
DEFAULT_RETRIEVAL_K = 4
SNIPPET_LENGTH = 200

SYSTEM_PROMPT = """You are a Government Scheme Navigator assistant.
Answer the user's question using ONLY the provided context from a single scheme document.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, respond with exactly:
  "{not_in_document_phrase}"
- When you can answer, mention relevant page numbers from the context when helpful.
- Be concise and factual.{summary_hint}"""

HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer:"""


@dataclass(frozen=True)
class SourceCitation:
    id: int
    page: int
    snippet: str


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SourceCitation]
    query_type: str = "single_fact"
    sub_queries: list[str] = field(default_factory=list)
    rerank_log: list[RerankEntry] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


def _format_context(scored: list[ScoredChunk]) -> str:
    blocks: list[str] = []
    for item in scored:
        chunk = item.chunk
        blocks.append(
            f"[Chunk {chunk.chunk_index} | page {chunk.page_number}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def _to_citations(
    scored: list[ScoredChunk], snippet_len: int = SNIPPET_LENGTH
) -> list[SourceCitation]:
    return [
        SourceCitation(
            id=item.chunk.chunk_index,
            page=item.chunk.page_number,
            snippet=item.chunk.text[:snippet_len],
        )
        for item in scored
    ]


def _merge_scored_chunks(chunks: list[ScoredChunk], k: int) -> list[ScoredChunk]:
    best_by_index: dict[int, ScoredChunk] = {}
    for item in chunks:
        chunk_index = item.chunk.chunk_index
        existing = best_by_index.get(chunk_index)
        if existing is None or item.score > existing.score:
            best_by_index[chunk_index] = item
    merged = sorted(best_by_index.values(), key=lambda item: item.score, reverse=True)
    return merged[:k]


def _retrieve_multi(
    vector_index: VectorIndex, sub_queries: list[str], k: int
) -> list[ScoredChunk]:
    all_scored: list[ScoredChunk] = []
    for sub_query in sub_queries:
        all_scored.extend(vector_index.query_index(sub_query, k=k))
    return _merge_scored_chunks(all_scored, k)


def _summary_hint(query_type: str) -> str:
    if query_type == "summarization":
        return "\n- The user wants a concise summary drawn only from the context."
    return ""


def create_rag_chain(
    vector_index: VectorIndex,
    llm: BaseChatModel,
    k: int = DEFAULT_RETRIEVAL_K,
) -> Runnable:
    def retrieve_step(inputs: dict[str, str]) -> dict[str, object]:
        question = inputs["question"]
        route = route_query(question, llm)
        scored = _retrieve_multi(vector_index, route.sub_queries, k=RERANK_TOP_N)
        reranked, rerank_log = rerank_chunks(question, scored, top_k=k)
        return {
            "question": question,
            "context": _format_context(reranked),
            "contexts": [item.chunk.text for item in reranked],
            "sources": _to_citations(reranked),
            "query_type": route.query_type,
            "sub_queries": route.sub_queries,
            "rerank_log": rerank_log,
        }

    def build_answer_chain(query_type: str) -> Runnable:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT.format(
                        not_in_document_phrase=NOT_IN_DOCUMENT_PHRASE,
                        summary_hint=_summary_hint(query_type),
                    ),
                ),
                ("human", HUMAN_PROMPT),
            ]
        )
        return (
            RunnableLambda(
                lambda state: {
                    "question": state["question"],
                    "context": state["context"],
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def answer_step(state: dict[str, object]) -> str:
        chain = build_answer_chain(str(state["query_type"]))
        return chain.invoke(state)

    return (
        RunnableLambda(retrieve_step)
        | RunnablePassthrough.assign(answer_text=RunnableLambda(answer_step))
        | RunnableLambda(
            lambda state: RAGAnswer(
                answer=state["answer_text"],
                sources=state["sources"],
                query_type=state["query_type"],
                sub_queries=state["sub_queries"],
                rerank_log=state["rerank_log"],
                contexts=state["contexts"],
            )
        )
    )


def ask_question(
    question: str,
    vector_index: VectorIndex,
    llm: BaseChatModel | None = None,
    k: int = DEFAULT_RETRIEVAL_K,
) -> RAGAnswer:
    chain = create_rag_chain(vector_index, llm or get_llm(), k=k)
    return chain.invoke({"question": question})
