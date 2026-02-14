from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from langchain_core.documents import Document

from cse_compass.ingestion.embeddings import get_embedding_model
from cse_compass.rag.vectorstore import get_chroma_vectorstore


def retrieve(
    query: str,
    persist_dir: Path = Path("data/vectorstore/chroma_db"),
    filters: Optional[Dict[str, Any]] = None,
    k: int = 5,
) -> List[Document]:
    embeddings = get_embedding_model()
    vs = get_chroma_vectorstore(persist_dir, embeddings)

    # Pass None when no filter (Chroma can error with empty dict in some versions)
    where = None
    if filters:
        where = {k: v for k, v in filters.items() if v is not None}
        if not where:
            where = None

    return vs.similarity_search(query, k=k, filter=where)
