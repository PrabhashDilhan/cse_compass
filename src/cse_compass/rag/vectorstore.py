from __future__ import annotations

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_chroma_vectorstore(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="cse_financial_reports",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
