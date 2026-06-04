"""Diagnose: file size and chunk count per raw file."""
from __future__ import annotations

from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_RAW_DIR
from ingest_local import load_local_documents


def main() -> None:
    # Per-file disk size
    print(f"{'File':<45} {'Bytes':>8}  {'Lines':>6}")
    print("-" * 65)
    for path in sorted(DATA_RAW_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        lines = text.count("\n")
        print(f"{path.name:<45} {size:>8}  {lines:>6}")

    print()
    print("=" * 65)
    print("Chunks generated per file (after splitting):")
    print("=" * 65)

    docs = load_local_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    counts = Counter(c.metadata.get("file_name", "?") for c in chunks)
    print(f"{'File':<45} {'Chunks':>8}")
    print("-" * 55)
    for name, count in sorted(counts.items()):
        marker = "  ⚠️" if count <= 2 else ""
        print(f"{name:<45} {count:>8}{marker}")

    print(f"\nTotal chunks: {sum(counts.values())}")

    # Specifically inspect the points file
    print("\n" + "=" * 65)
    print("First 500 chars of points_test_explainer.txt:")
    print("=" * 65)
    points_path = DATA_RAW_DIR / "points_test_explainer.txt"
    if points_path.exists():
        text = points_path.read_text(encoding="utf-8", errors="replace")
        print(text[:500])
        print(f"\n[Total length: {len(text)} chars]")
    else:
        print("FILE NOT FOUND")


if __name__ == "__main__":
    main()