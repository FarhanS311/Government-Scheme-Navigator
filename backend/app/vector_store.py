"""FAISS vector index build, persistence, and similarity search over scheme chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from app.ingestion import Chunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path("data/index")
INDEX_NAME = "index"

_embeddings: HuggingFaceEmbeddings | None = None


@dataclass(frozen=True)
class ScoredChunk:
    """A ``Chunk`` paired with its retrieval similarity score."""

    chunk: Chunk
    score: float


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load and cache the sentence-transformer embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    """Convert internal chunks into LangChain documents for FAISS indexing."""
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk in chunks
    ]


def _document_to_chunk(document: Document) -> Chunk:
    """Rebuild a ``Chunk`` from a FAISS-stored LangChain document."""
    metadata = document.metadata
    return Chunk(
        text=document.page_content,
        source=str(metadata["source"]),
        page_number=int(metadata["page_number"]),
        chunk_index=int(metadata["chunk_index"]),
    )


def _index_files_exist(persist_dir: Path) -> bool:
    """Return True when both FAISS artifact files are present on disk."""
    return (persist_dir / f"{INDEX_NAME}.faiss").exists() and (
        persist_dir / f"{INDEX_NAME}.pkl"
    ).exists()


class VectorIndex:
    """Thin wrapper around a loaded FAISS store for top-k similarity search."""

    def __init__(self, store: FAISS) -> None:
        """Store the underlying LangChain FAISS instance."""
        self._store = store

    def query_index(self, question: str, k: int = 4) -> list[ScoredChunk]:
        """Run inner-product similarity search and return chunks sorted by score."""
        if k < 1:
            raise ValueError("k must be at least 1")

        results = self._store.similarity_search_with_score(question, k=k)
        scored = [
            ScoredChunk(chunk=_document_to_chunk(doc), score=float(score))
            for doc, score in results
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored


def build_index(chunks: list[Chunk], persist_dir: Path = DEFAULT_INDEX_DIR) -> None:
    """Embed chunks and write a new FAISS index to ``persist_dir``."""
    if not chunks:
        raise ValueError("cannot build index from empty chunk list")

    persist_dir.mkdir(parents=True, exist_ok=True)
    documents = chunks_to_documents(chunks)
    store = FAISS.from_documents(
        documents,
        _get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    store.save_local(str(persist_dir), index_name=INDEX_NAME)


def load_index(persist_dir: Path = DEFAULT_INDEX_DIR) -> VectorIndex:
    """Load a persisted FAISS index from disk."""
    if not persist_dir.exists():
        raise FileNotFoundError(f"index directory not found: {persist_dir}")
    if not _index_files_exist(persist_dir):
        raise FileNotFoundError(
            f"index files not found in {persist_dir}; run build_index first"
        )

    store = FAISS.load_local(
        str(persist_dir),
        _get_embeddings(),
        index_name=INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    return VectorIndex(store)