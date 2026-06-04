"""Debug: see what hybrid retrieval returns for a specific query."""
from __future__ import annotations

from hybrid_retriever import get_bm25_retriever, get_vector_retriever, hybrid_retrieve


def show(docs, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", "?")
        file_name = d.metadata.get("file_name", "?")
        chunk_id = d.metadata.get("chunk_id", "?")
        preview = d.page_content[:150].replace("\n", " ")
        print(f"  {i}. [{file_name} | chunk {chunk_id}] {title}")
        print(f"     {preview}...")


def main() -> None:
    queries = [
        "How many points would I get?",
        "points test breakdown",
        "skilled migration points age English",
    ]

    for q in queries:
        print(f"\n\n{'#' * 70}")
        print(f"# QUERY: {q}")
        print(f"{'#' * 70}")

        bm25 = get_bm25_retriever().invoke(q)
        show(bm25[:6], "BM25 (lexical) — top 6")

        vector = get_vector_retriever().invoke(q)
        show(vector[:6], "Dense (embeddings) — top 6")

        fused = hybrid_retrieve(q)
        show(fused, "RRF fused — final top 7")


if __name__ == "__main__":
    main()