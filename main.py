"""Visa Navigator — entry point with usage hints."""
from __future__ import annotations

import sys

USAGE = """
AI Visa Navigator — RAG over Australian student visa info

Usage:
  cd app && python scrape_visa_page.py    # (Re)scrape source pages with Playwright
  cd app && python build_index.py         # Build the Chroma vector index from data/raw/
  cd app && python query.py               # Start the CLI chatbot
  streamlit run streamlit_app.py          # Launch the web UI (Phase 2)

First-time setup:
  1. cp .env.example .env  (and add your OPENAI_API_KEY)
  2. pip install -r requirements.txt
  3. playwright install chromium   # only if you plan to (re)scrape
  4. cd app && python build_index.py
"""


def main() -> None:
    print(USAGE)
    sys.exit(0 if len(sys.argv) == 1 else 1)


if __name__ == "__main__":
    main()