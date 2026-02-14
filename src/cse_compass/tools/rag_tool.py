from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from cse_compass.rag.retriever import retrieve


def _rag_search(
    query: str,
    ticker: Optional[str] = None,
    k: int = 5,
) -> str:
    """Search RAG for financial report excerpts."""
    filters = {"ticker": ticker} if ticker else None
    docs = retrieve(query, filters=filters, k=k)
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata.get("ticker", "") or d.metadata.get("source_file", "")
        parts.append(f"[{i}] {d.page_content[:800]}...\n(Source: {meta})")
    return "\n\n".join(parts) if parts else "No relevant documents found."


def get_rag_search_tool():
    return StructuredTool.from_function(
        name="rag_search",
        description="Search the RAG database for financial report excerpts. Use ticker (e.g. DFCC.N0000) to filter by company.",
        func=_rag_search,
    )
