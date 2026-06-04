"""Interactive CLI chatbot. Wraps rag_pipeline.answer_question in a REPL."""
from __future__ import annotations

from rag_pipeline import answer_question, get_llm


def main() -> None:
    llm = get_llm()

    print("Visa RAG Assistant — Hybrid Retrieval (BM25 + Dense + RRF)")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            print("Please enter a question.\n")
            continue

        result = answer_question(question, llm=llm)

        print("\nANSWER:\n")
        print(result.answer)

        print("\nRETRIEVED SOURCES:")
        for d in result.sources:
            print(
                f"  - {d.metadata.get('title')} "
                f"| chunk {d.metadata.get('chunk_id')} "
                f"| {d.metadata.get('source')}"
            )
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()