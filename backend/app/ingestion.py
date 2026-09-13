from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 300


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page_number: int
    chunk_index: int


def ingest_pdf(path: Path | str) -> list[Chunk]:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("path must be a pdf file")

    documents = PyPDFLoader(str(pdf_path)).load()
    nonempty = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_documents(nonempty)

    source = pdf_path.name
    chunks: list[Chunk] = []
    for index, document in enumerate(splits):
        page = document.metadata.get("page", 0)
        chunks.append(
            Chunk(
                text=document.page_content,
                source=source,
                page_number=int(page) + 1,
                chunk_index=index,
            )
        )
    return chunks
