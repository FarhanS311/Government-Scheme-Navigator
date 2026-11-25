from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.config import get_llm
from app.query_router import route_query
from app.reranker import (
    RERANK_TOP_N,
    RerankEntry,
    filter_reranked_by_score,
    rerank_chunks,
)
from app.schemas import ChatTurn
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
{conversation_block}
Question: {question}

Answer:"""

REWRITE_PROMPT = """Given the conversation and the user's latest follow-up question, write ONE standalone search query that captures what the user wants to know from a government scheme PDF.

Output only the query text, with no quotes or explanation.

Conversation:
{history}

Latest question: {question}

Standalone search query:"""


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


@dataclass(frozen=True)
class PreparedRAG:
    question: str
    search_question: str
    history: list[ChatTurn]
    query_type: str
    sub_queries: list[str]
    reranked: list[ScoredChunk]
    rerank_log: list[RerankEntry]
    abstain: bool
    context: str
    contexts: list[str]
    sources: list[SourceCitation]


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


def _format_history(history: list[ChatTurn]) -> str:
    lines: list[str] = []
    for turn in history:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def _conversation_block(history: list[ChatTurn]) -> str:
    if not history:
        return ""
    return f"\nPrior conversation:\n{_format_history(history)}\n"


def rewrite_follow_up(
    question: str,
    history: list[ChatTurn],
    llm: BaseChatModel,
) -> str:
    if not history:
        return question

    prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        rewritten = chain.invoke(
            {"history": _format_history(history), "question": question}
        )
        cleaned = rewritten.strip()
        return cleaned or question
    except Exception:
        return question


def build_answer_chain(query_type: str, llm: BaseChatModel) -> Runnable:
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
                "conversation_block": state["conversation_block"],
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def prepare_rag_state(
    question: str,
    vector_index: VectorIndex,
    llm: BaseChatModel,
    k: int = DEFAULT_RETRIEVAL_K,
    history: list[ChatTurn] | None = None,
) -> PreparedRAG:
    prior = history or []
    search_question = rewrite_follow_up(question.strip(), prior, llm)
    route = route_query(search_question, llm)
    scored = _retrieve_multi(vector_index, route.sub_queries, k=RERANK_TOP_N)
    reranked, rerank_log = rerank_chunks(search_question, scored, top_k=k)
    reranked = filter_reranked_by_score(reranked)

    if not reranked:
        return PreparedRAG(
            question=question.strip(),
            search_question=search_question,
            history=prior,
            query_type=route.query_type,
            sub_queries=route.sub_queries,
            reranked=[],
            rerank_log=rerank_log,
            abstain=True,
            context="",
            contexts=[],
            sources=[],
        )

    return PreparedRAG(
        question=question.strip(),
        search_question=search_question,
        history=prior,
        query_type=route.query_type,
        sub_queries=route.sub_queries,
        reranked=reranked,
        rerank_log=rerank_log,
        abstain=False,
        context=_format_context(reranked),
        contexts=[item.chunk.text for item in reranked],
        sources=_to_citations(reranked),
    )


def _answer_invoke_state(prepared: PreparedRAG) -> dict[str, str]:
    return {
        "question": prepared.question,
        "context": prepared.context,
        "conversation_block": _conversation_block(prepared.history),
    }


def generate_answer(prepared: PreparedRAG, llm: BaseChatModel) -> str:
    if prepared.abstain:
        return NOT_IN_DOCUMENT_PHRASE
    chain = build_answer_chain(prepared.query_type, llm)
    return chain.invoke(_answer_invoke_state(prepared))


async def stream_answer_tokens(
    prepared: PreparedRAG,
    llm: BaseChatModel,
) -> AsyncIterator[str]:
    if prepared.abstain:
        yield NOT_IN_DOCUMENT_PHRASE
        return

    chain = build_answer_chain(prepared.query_type, llm)
    async for token in chain.astream(_answer_invoke_state(prepared)):
        if token:
            yield token


def prepared_to_rag_answer(prepared: PreparedRAG, answer: str) -> RAGAnswer:
    return RAGAnswer(
        answer=answer,
        sources=prepared.sources,
        query_type=prepared.query_type,
        sub_queries=prepared.sub_queries,
        rerank_log=prepared.rerank_log,
        contexts=prepared.contexts,
    )


def create_rag_chain(
    vector_index: VectorIndex,
    llm: BaseChatModel,
    k: int = DEFAULT_RETRIEVAL_K,
) -> Runnable:
    def run(inputs: dict[str, str]) -> RAGAnswer:
        question = inputs["question"]
        prepared = prepare_rag_state(question, vector_index, llm, k=k)
        answer = generate_answer(prepared, llm)
        return prepared_to_rag_answer(prepared, answer)

    return RunnableLambda(run)


def ask_question(
    question: str,
    vector_index: VectorIndex,
    llm: BaseChatModel | None = None,
    k: int = DEFAULT_RETRIEVAL_K,
    history: list[ChatTurn] | None = None,
) -> RAGAnswer:
    model = llm or get_llm()
    prepared = prepare_rag_state(question, vector_index, model, k=k, history=history)
    answer = generate_answer(prepared, model)
    return prepared_to_rag_answer(prepared, answer)


def iter_answer_tokens(
    prepared: PreparedRAG,
    llm: BaseChatModel,
) -> Iterator[str]:
    if prepared.abstain:
        yield NOT_IN_DOCUMENT_PHRASE
        return
    chain = build_answer_chain(prepared.query_type, llm)
    yield from chain.stream(_answer_invoke_state(prepared))