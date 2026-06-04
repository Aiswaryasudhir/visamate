"""Live-scrape URLs via WebBaseLoader (alternative to Playwright scraper)."""
from __future__ import annotations

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from config import DATA_PROCESSED_DIR, SCRAPE_URLS
from utils import clean_html_to_text


def load_web_documents(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(web_paths=tuple(urls))
    docs = loader.load()

    cleaned_docs: list[Document] = []
    for doc in docs:
        cleaned_text = clean_html_to_text(doc.page_content)
        cleaned_docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "title": doc.metadata.get("title", ""),
                    "doc_type": "visa_webpage",
                },
            )
        )

    return cleaned_docs


if __name__ == "__main__":
    urls = list(SCRAPE_URLS.values())
    print(f"Loading {len(urls)} URL(s)...")
    for url in urls:
        print(f"  - {url}")

    docs = load_web_documents(urls)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for i, doc in enumerate(docs, start=1):
        output_path = DATA_PROCESSED_DIR / f"doc_{i}.txt"
        output_path.write_text(doc.page_content, encoding="utf-8")
        print(f"Saved: {output_path}")
        print(doc.metadata)
        print(doc.page_content[:300])
        print("-" * 80)