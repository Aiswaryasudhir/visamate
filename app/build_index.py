"""Chunk local documents and embed them into a fresh Chroma collection."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    validate_environment,
)
from ingest_local import load_local_documents


def main() -> None:
    validate_environment()

    docs = load_local_documents()
    if not docs:
        raise ValueError("No local documents found in data/raw/")

    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"Deleted existing Chroma directory: {CHROMA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # Prepend the title to EACH chunk (post-split) so every chunk
        # carries strong topic keywords for retrieval. The title only
        # survives in chunk 0 if prepended pre-split.
        title = chunk.metadata.get("title", "")
        if title:
            chunk.page_content = f"[{title}]\n{chunk.page_content}"

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Indexed {len(chunks)} chunks into Chroma.")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Persist dir: {CHROMA_DIR}")


if __name__ == "__main__":
    main()