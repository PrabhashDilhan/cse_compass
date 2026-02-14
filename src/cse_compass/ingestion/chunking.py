from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Splits page-documents into smaller chunks.
    Keeps original metadata (ticker/year/page/source_file), and adds chunk index.
    Table documents (content_type='table') are kept intact, not split.
    """
    text_docs = [d for d in docs if d.metadata.get("content_type") != "table"]
    table_docs = [d for d in docs if d.metadata.get("content_type") == "table"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_documents(text_docs) if text_docs else []

    # add chunk_in_page for text chunks
    counter = {}
    for d in text_chunks:
        key = (d.metadata.get("source_file"), d.metadata.get("page"))
        counter[key] = counter.get(key, 0) + 1
        d.metadata["chunk_in_page"] = counter[key]

    # tables stay as single chunks, chunk_in_page = 0 (not used for ID)
    for d in table_docs:
        d.metadata["chunk_in_page"] = 0

    return text_chunks + table_docs
