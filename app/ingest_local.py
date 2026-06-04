"""Load local .txt files from data/raw/ into LangChain Documents."""
from __future__ import annotations

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from config import DATA_RAW_DIR, SOURCE_MAP


def load_local_documents() -> list[Document]:
    documents: list[Document] = []

    for file_path in sorted(DATA_RAW_DIR.glob("*.txt")):
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            title = file_path.stem.replace("_", " ").title()
            doc.metadata.update(
                {
                    "source": SOURCE_MAP.get(file_path.name, "unknown_source"),
                    "title": title,
                    "doc_type": "official_visa_policy",
                    "file_name": file_path.name,
                }
            )
            documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = load_local_documents()
    print(f"Loaded {len(docs)} local document(s).")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content[:300])
        print("-" * 80)