"""Ingest processed reports (text.md, tables.json, metadata.json) into Chroma."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document

from cse_compass.ingestion.loaders import iter_processed_report_dirs, load_processed_report
from cse_compass.ingestion.metadata import infer_metadata_from_processed_path
from cse_compass.ingestion.chunking import chunk_documents
from cse_compass.ingestion.embeddings import get_embedding_model
from cse_compass.rag.vectorstore import get_chroma_vectorstore
from dotenv import load_dotenv

load_dotenv()


def save_chunks_jsonl(chunks: List[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in chunks:
            rec = {
                "page_content": d.page_content,
                "metadata": d.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ingest(
    input_dir: Path,
    persist_dir: Path,
    cache_chunks_dir: Path | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> None:
    """
    Ingest processed reports (text + tables) into Chroma.
    Uses combined.md/text.md for narrative and tables.json for financial tables.
    """
    embeddings = get_embedding_model()
    vs = get_chroma_vectorstore(persist_dir, embeddings)

    report_dirs = sorted(set(iter_processed_report_dirs(input_dir)))
    if not report_dirs:
        raise RuntimeError(f"No processed reports found under: {input_dir}")

    total_added = 0

    for report_dir in tqdm(report_dirs, desc="Ingesting processed reports"):
        base_meta = infer_metadata_from_processed_path(report_dir)
        docs = load_processed_report(report_dir, base_meta)
        if not docs:
            continue

        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        ids = []
        for d in chunks:
            source = d.metadata.get("source_file")
            page = d.metadata.get("page")
            content_type = d.metadata.get("content_type", "text")
            c = d.metadata.get("chunk_in_page")
            tbl = d.metadata.get("table_index")
            if content_type == "table" and tbl is not None:
                ids.append(f"{source}::p{page}::t{tbl}")
            else:
                ids.append(f"{source}::p{page}::c{c}")

        if cache_chunks_dir is not None:
            save_chunks_jsonl(
                chunks,
                cache_chunks_dir / f"{report_dir.name}.chunks.jsonl",
            )

        # Chroma rejects None in metadata - strip before add
        for d in chunks:
            d.metadata = {k: v for k, v in d.metadata.items() if v is not None}

        vs.add_documents(chunks, ids=ids)
        total_added += len(chunks)

    print(f"✅ Done. Added/updated ~{total_added} chunks into: {persist_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to processed_reports root (e.g. data/raw/cse/processed_reports)",
    )
    parser.add_argument(
        "--persist",
        required=True,
        help="Chroma persist directory (data/vectorstore/chroma_db)",
    )
    parser.add_argument(
        "--cache-chunks",
        default=None,
        help="Optional: cache JSONL chunks directory",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)

    args = parser.parse_args()

    ingest(
        input_dir=Path(args.input),
        persist_dir=Path(args.persist),
        cache_chunks_dir=Path(args.cache_chunks) if args.cache_chunks else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
