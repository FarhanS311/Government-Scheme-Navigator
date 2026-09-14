from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.config import get_llm
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
- Be concise and factual."""

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


def _retrieve(
    vector_index: VectorIndex, question: str, k: int
) -> dict[str, object]:
    scored = vector_index.query_index(question, k=k)
    return {
        "question": question,
        "context": _format_context(scored),
        "sources": _to_citations(scored),
    }


def create_rag_chain(
    vector_index: VectorIndex,
    llm: BaseChatModel,
    k: int = DEFAULT_RETRIEVAL_K,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT.format(not_in_document_phrase=NOT_IN_DOCUMENT_PHRASE),
            ),
            ("human", HUMAN_PROMPT),
        ]
    )

    def retrieve_step(inputs: dict[str, str]) -> dict[str, object]:
        return _retrieve(vector_index, inputs["question"], k)

    answer_chain = (
        RunnableLambda(lambda state: {"question": state["question"], "context": state["context"]})
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnableLambda(retrieve_step)
        | RunnablePassthrough.assign(answer_text=answer_chain)
        | RunnableLambda(
            lambda state: RAGAnswer(
                answer=state["answer_text"],
                sources=state["sources"],
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
