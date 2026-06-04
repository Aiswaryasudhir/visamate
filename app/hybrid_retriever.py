"""Hybrid retrieval: BM25 (lexical) + Chroma (dense) fused with Reciprocal Rank Fusion.

Why hybrid?
    BM25 catches exact-term matches (visa subclass numbers, condition codes).
    Dense embeddings catch semantic paraphrases ("can my partner work?" vs "secondary applicant rights").
    RRF combines both ranked lists without requiring score normalization.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    BM25_K,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    FINAL_K,
    RRF_K_CONSTANT,
    VECTOR_K,
)
from ingest_local import load_local_documents


def _make_doc_id(doc: Document) -> str:
    """Stable identifier for de-duplication across BM25 and vector results."""
    return "|".join(
        [
            doc.metadata.get("source", ""),
            doc.metadata.get("file_name", ""),
            str(doc.metadata.get("chunk_id", "")),
            doc.page_content[:120],
        ]
    )


def _get_chunked_documents() -> List[Document]:
    docs = load_local_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        title = chunk.metadata.get("title", "")
        if title:
            chunk.page_content = f"[{title}]\n{chunk.page_content}"
    return chunks


@lru_cache(maxsize=1)
def get_bm25_retriever() -> BM25Retriever:
    """Cached: load + tokenize once per process, not per query."""
    chunks = _get_chunked_documents()
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = BM25_K
    return retriever


@lru_cache(maxsize=1)
def get_vector_retriever():
    """Cached: open Chroma connection once per process."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": VECTOR_K})


def reciprocal_rank_fusion(
    bm25_docs: List[Document],
    vector_docs: List[Document],
    final_k: int = FINAL_K,
    k_constant: int = RRF_K_CONSTANT,
) -> List[Document]:
    """Combine two ranked lists into one using RRF (Cormack et al., 2009).

    Score for doc d = sum over rankings of 1 / (k_constant + rank_in_that_list).
    Higher rank => higher score. Robust to differing score scales between retrievers.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_lookup: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = _make_doc_id(doc)
        scores[doc_id] += 1.0 / (k_constant + rank)
        doc_lookup[doc_id] = doc

    for rank, doc in enumerate(vector_docs, start=1):
        doc_id = _make_doc_id(doc)
        scores[doc_id] += 1.0 / (k_constant + rank)
        doc_lookup[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[doc_id] for doc_id, _ in ranked[:final_k]]


def hybrid_retrieve(query: str) -> List[Document]:
    """Run both retrievers in parallel-ish and fuse results."""
    bm25_docs = get_bm25_retriever().invoke(query)
    vector_docs = get_vector_retriever().invoke(query)
    return reciprocal_rank_fusion(bm25_docs, vector_docs)