from __future__ import annotations

import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document

from cse_compass.ingestion.loaders import iter_report_files, load_pdf_documents
from cse_compass.ingestion.metadata import infer_metadata_from_path
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
    embeddings = get_embedding_model()
    vs = get_chroma_vectorstore(persist_dir, embeddings)

    pdf_files = sorted(list(iter_report_files(input_dir)))
    if not pdf_files:
        raise RuntimeError(f"No PDFs found under: {input_dir}")

    total_added = 0

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
        md = infer_metadata_from_path(pdf_path)
        page_docs = load_pdf_documents(pdf_path, base_metadata=md)

        # Optionally: drop empty pages
        page_docs = [d for d in page_docs if d.page_content and d.page_content.strip()]

        chunks = chunk_documents(page_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Add a stable ID to each chunk (helps dedupe and deterministic updates)
        ids = []
        for d in chunks:
            source = d.metadata.get("source_file")
            page = d.metadata.get("page")
            c = d.metadata.get("chunk_in_page")
            ids.append(f"{source}::p{page}::c{c}")

        if cache_chunks_dir is not None:
            save_chunks_jsonl(chunks, cache_chunks_dir / f"{pdf_path.stem}.chunks.jsonl")

        # Chroma rejects None in metadata - strip before add
        for d in chunks:
            d.metadata = {k: v for k, v in d.metadata.items() if v is not None}

        # Upsert into Chroma
        vs.add_documents(chunks, ids=ids)
        total_added += len(chunks)

    # Chroma auto-persists when using persist_directory
    print(f"✅ Done. Added/updated ~{total_added} chunks into: {persist_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to PDFs root (data/raw/...)")
    parser.add_argument("--persist", required=True, help="Chroma persist directory (data/vectorstore/chroma_db)")
    parser.add_argument("--cache-chunks", default=None, help="Optional: cache JSONL chunks directory (data/processed/chunks)")
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
