from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def iter_processed_report_dirs(root: Path) -> Iterable[Path]:
    """Yield report directories under processed_reports (each has metadata.json, text.md, tables.json)."""
    for p in root.rglob("metadata.json"):
        if p.is_file():
            yield p.parent


def load_processed_report(report_dir: Path, base_metadata: Dict[str, Any]) -> List[Document]:
    """
    Load a processed report from text.md/combined.md, tables.json, and metadata.json.
    Returns Documents for main text (chunked by page) and for each significant table.
    """
    docs: List[Document] = []

    # Load metadata
    meta_path = report_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            proc_meta = json.load(f)
        base_metadata = {**base_metadata, "tables_count": proc_meta.get("tables_count", 0)}

    # Load main text (prefer combined.md, fallback to text.md)
    text_path = report_dir / "combined.md"
    if not text_path.exists():
        text_path = report_dir / "text.md"
    if not text_path.exists():
        return docs

    text_content = text_path.read_text(encoding="utf-8")
    # Strip the header block (lines before "## Page" or "# Extracted Text")
    text_content = re.sub(r"^# [^\n]+\n(\*\*[^*]+\*\*[^\n]*\n)*---*\n*", "", text_content)
    text_content = re.sub(r"^# Extracted Text\s*\n+", "", text_content)

    # Split by page sections (## Page N) - re.split includes capture groups in result
    page_sections = re.split(r"## Page (\d+)\n", text_content)
    for i in range(1, len(page_sections) - 1, 2):
        page_num = int(page_sections[i])
        page_text = page_sections[i + 1].strip()
        if not page_text:
            continue
        doc = Document(
            page_content=page_text,
            metadata={
                **base_metadata,
                "page": page_num,
                "source_file": report_dir.name,
                "content_type": "text",
            },
        )
        docs.append(doc)

    # Load tables
    tables_path = report_dir / "tables.json"
    if tables_path.exists():
        with tables_path.open(encoding="utf-8") as f:
            tables = json.load(f)
        for t in tables:
            md = t.get("markdown", "").strip()
            if not md or len(md) < 30:
                continue
            rows = t.get("rows", 0)
            if rows < 2:
                continue
            doc = Document(
                page_content=f"Table (page {t['page']}):\n\n{md}",
                metadata={
                    **base_metadata,
                    "page": t["page"],
                    "table_index": t.get("table_index", 0),
                    "source_file": report_dir.name,
                    "content_type": "table",
                    "table_rows": rows,
                    "table_columns": t.get("columns", 0),
                },
            )
            docs.append(doc)

    return docs


def load_pdf_documents(pdf_path: Path, base_metadata: Dict[str, Any]) -> List[Document]:
    """
    Loads a PDF into a list of Documents, typically one per page, depending on loader.
    Each Document gets merged metadata.
    """
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    for d in docs:
        # Merge metadata (document-level) into page docs
        d.metadata = {**d.metadata, **base_metadata}
        d.metadata["source_file"] = pdf_path.name
        # PyPDFLoader adds "page" index in metadata for citations
        # Keep it for page-based citations later
    return docs


def iter_report_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p
