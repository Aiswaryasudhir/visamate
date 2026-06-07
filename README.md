---
title: VisaMate
emoji: 🌿
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.36.0
app_file: streamlit_app.py
pinned: false
license: mit
---
# 🌿 VisaMate

> A conversational Retrieval-Augmented Generation (RAG) system grounded in official Australian government sources, designed to help international students navigate the path from student visa to permanent residency.

**[Live demo](https://huggingface.co/spaces/Aiswarya2501/visamate)** · **[GitHub](https://github.com/Aiswaryasudhir/visamate)**

---

## Table of contents

- [About](#about)
- [Demo](#demo)
- [System architecture](#system-architecture)
- [Data pipeline](#data-pipeline)
- [Retrieval](#retrieval)
- [Conversational layer](#conversational-layer)
- [Generation](#generation)
- [Multi-provider LLM abstraction](#multi-provider-llm-abstraction)
- [Automated content refresh](#automated-content-refresh)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Run it locally](#run-it-locally)
- [Configuration reference](#configuration-reference)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [What's next](#whats-next)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## About

The Australian visa system changes constantly. Subclass 500 application fees moved from AUD $710 to $2,000 in just over a year. The 485 visa was restructured in 2024. The Genuine Temporary Entrant (GTE) test was replaced by the Genuine Student (GS) requirement on 23 March 2024. The Temporary Skill Shortage 482 was renamed Skills in Demand on 7 December 2024 with three new streams. Most international students piece their understanding together from forum threads, agent blogs, and a maze of official Home Affairs pages.

VisaMate is a domain-specific RAG system grounded in official Australian government sources. It covers:

- **Subclass 500** — fees, documents, Genuine Student requirement, conditions, dependents
- **Subclass 485** — Post-Higher Education and Post-Vocational Education Work streams, stay durations, eligibility
- **PR pathways** — 189 (Skilled Independent), 190 (Skilled Nominated), 491 (Skilled Work Regional), 482 → 186 (employer-sponsored)
- **Points test** — full breakdown across all 12 categories with worked examples
- **State nomination** — what each state prioritises, 2025–26 allocations

The assistant captures the user's situation — field of study, English level, age, regional openness, work experience — through natural conversation, and conditions both retrieval and generation on that profile across the session.

> ⚠️ Educational tool. Always verify visa decisions with [Home Affairs](https://immi.homeaffairs.gov.au/) or a MARA-registered migration agent.

---

## Demo

**Conversational context capture.** VisaMate detects when a question requires user context, asks for everything it needs in one consolidated follow-up, then remembers the answers for the rest of the session.

![Conversational follow-up](docs/screenshots/follow-up.png)

**Grounded answers with source citations.** Every answer cites the documents that produced it, with direct links to the official source.

![Grounded answer with citations](docs/screenshots/grounded-answer.png)

**Sidebar profile panel.** The system's understanding of the user is visible and editable, not hidden in chat history.

![Sidebar profile](docs/screenshots/profile-sidebar.png)

---

## System architecture

```
                    ┌──────────────────────────────┐
                    │       Streamlit UI           │
                    │  (chat + sidebar profile)    │
                    └──────────────┬───────────────┘
                                   │ user message
                                   ▼
                    ┌──────────────────────────────┐
                    │  Profile attribute extractor │
                    │       (LLM, JSON output)     │
                    └──────────────┬───────────────┘
                                   │ updates
                                   ▼
                    ┌──────────────────────────────┐
                    │   UserProfile (session)      │
                    │  name, field_of_study,       │
                    │  english_level, age, ...     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Routing classifier      │
                    │       (LLM, JSON output)     │
                    │   "answer" or "ask"?         │
                    └──────┬───────────────┬───────┘
                           │               │
                ┌──────────▼──────┐  ┌─────▼────────┐
                │  Follow-up      │  │  Hybrid      │
                │  generator      │  │  retriever   │
                │  (LLM)          │  │              │
                └──────────┬──────┘  └─────┬────────┘
                           │               │
                           │      ┌────────┴────────┐
                           │      ▼                 ▼
                           │  ┌────────┐      ┌─────────┐
                           │  │ BM25   │      │ Chroma  │
                           │  │(rank-  │      │ (dense  │
                           │  │ bm25)  │      │ embeds) │
                           │  └───┬────┘      └────┬────┘
                           │      │   top-K each   │
                           │      └────────┬───────┘
                           │               ▼
                           │      ┌────────────────┐
                           │      │ RRF fusion     │
                           │      │ (final top-N)  │
                           │      └────────┬───────┘
                           │               ▼
                           │      ┌────────────────┐
                           │      │ Grounded       │
                           │      │ generator      │
                           │      │ (LLM + prompt) │
                           │      └────────┬───────┘
                           │               │
                           └───────┬───────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │   Streamlit response render  │
                    │   (answer + sources panel)   │
                    └──────────────────────────────┘

       Offline / scheduled:
       ┌────────────────────────────────────────────┐
       │ Playwright scraper ──► Curated .txt files  │
       │           │                                │
       │           ▼                                │
       │ Text splitter (RecursiveCharacterTextSplit)│
       │           │                                │
       │           ▼                                │
       │ Chunking + title prepending + chunk_id     │
       │           │                                │
       │           ▼                                │
       │ Embedding (OpenAI text-embedding-3-small)  │
       │           │                                │
       │           ▼                                │
       │ Chroma persistent store                    │
       └────────────────────────────────────────────┘

       Scheduled (GitHub Actions):
       weekly cron → re-scrape → rebuild index → commit → redeploy
```

Each turn flows: extract attributes from the user message → update profile → classify route → either generate a follow-up question or run hybrid retrieval + grounded generation. Profile updates trigger a Streamlit rerun so the sidebar reflects the new state immediately.

---

## Data pipeline

The corpus is built from 16 source documents — a mix of scraped and curated content.

### Scraping (Playwright)

Five Home Affairs and related pages are scraped using **Playwright**. Government pages use JS-rendered accordion sections (eligibility, conditions, fees, document checklists) that aren't visible to a simple HTTP fetcher. The scraper:

1. Launches a headless Chromium instance
2. Loads each URL and waits for the DOM
3. Clicks every collapsible element matching a list of selectors (`button[aria-expanded='false']`, `summary`, `.accordion-header`, etc.) so all sections expand into the visible text
4. Strips noisy DOM nodes (header, footer, nav, search widgets)
5. Extracts text from `article`, `main`, or `.content` selectors with a body fallback
6. Writes each page to `data/raw/<name>.txt`

Scraped sources include the 485 Post-Higher Education Work stream, 485 Post-Vocational Education Work stream, the Genuine Student requirement page, and the 485 changes log.

### Curated content

Some content scrapes poorly — fee tables rendered in JavaScript widgets, points test breakdowns with nested ranges, cross-cutting overviews that don't exist on any single page. These are written as plain-text reference documents:

- `visa_costs_2026.txt` — dated fee breakdown for 500, 485, dependents, ancillary costs
- `documents_required.txt` — consolidated document checklist including GS requirement
- `points_test_explainer.txt` — all 12 points categories with thresholds and worked example
- `pr_pathways_overview.txt` — the student → 485 → PR roadmap
- `state_nomination_overview.txt` — 2025–26 state allocations and priorities

Each file is dated and source-attributed, so when fees or rules change, the relevant document is the only thing that needs to be updated.

### Source mapping

Every file in `data/raw/` is mapped to an origin URL in `app/config.py` (`SOURCE_MAP`). This metadata follows each document through chunking and surfaces in the UI as citation links — so a user reading an answer about the 485 fee can click through to the actual Home Affairs page.

### Chunking

Documents are split with LangChain's `RecursiveCharacterTextSplitter`:

- **Chunk size:** 800 characters
- **Overlap:** 150 characters
- **Separators:** default (paragraph, line, sentence)

Before embedding, each chunk's content is prefixed with `[Document Title]` (see [Design decisions](#design-decisions) for why). Chunks are assigned a sequential `chunk_id` and inherit `source`, `title`, `doc_type`, and `file_name` metadata from the parent document.

The full corpus produces approximately 185 chunks.

---

## Retrieval

VisaMate uses **hybrid retrieval with Reciprocal Rank Fusion**, combining lexical and semantic search.

### BM25 (lexical)

Implemented via LangChain's `BM25Retriever` (backed by `rank_bm25`). Builds an in-memory index over the chunked corpus at startup. Strong for:

- Exact-term matches (visa subclass numbers: "189", "485", "subclass 500")
- Condition codes ("8104", "8202", "8501")
- Specific document references ("MLTSSL", "STSOL", "CSOL")
- Numeric matches ("$2,000", "65 points", "IELTS 7")

### Dense embeddings (semantic)

Documents are embedded with **OpenAI `text-embedding-3-small`** (1536 dimensions) and stored in **ChromaDB** with a persistent directory (`chroma_db/`). Strong for:

- Paraphrased questions ("can my partner work full-time" → "secondary applicant work conditions")
- Conceptual queries that don't share vocabulary with the source
- Multi-hop reasoning starters ("what happens after my course ends")

### Reciprocal Rank Fusion (RRF)

The two retrievers return overlapping but not identical ranked lists. To combine them without normalising heterogeneous score scales (BM25 scores and cosine similarities aren't comparable), the system uses **Reciprocal Rank Fusion** (Cormack et al., 2009):

```
score(d) = Σ over rankings of 1 / (k + rank_in_that_list)
```

Where `k = 60` (the standard constant from the original paper). Documents are de-duplicated by a stable composite ID before scoring.

The retrievers each return 6 candidates, RRF fuses them, and the top 7 documents are passed to the generator.

### Caching

Both retrievers are constructed once per process using `@lru_cache(maxsize=1)` — BM25 is tokenised once at first query, Chroma is opened once and reused. No rebuild on every question.

### Profile-aware query enrichment (conditional)

The retrieval query is augmented with profile attributes (occupation, state, "regional") only when the question is generic. A keyword gate inspects the question for topic anchors ("points", "fee", "documents", subclass numbers, English test names) — if any match, the raw question is used unmodified, because the topic keyword is the strongest retrieval signal and augmentation drowns it out. This was discovered empirically: augmenting "How many points would I get?" with profile keywords caused the points file to lose ranking to PR pathway files that also mentioned points incidentally.

---

## Conversational layer

The conversational layer turns a stateless RAG bot into a stateful agent. Three LLM-powered components.

### Attribute extractor (`user_profile.py`)

Every user message is passed through an LLM extractor that returns a JSON object of any attributes the user explicitly stated or strongly implied. Fields:

- `name`
- `field_of_study`
- `intended_occupation`
- `english_level` (Competent / Proficient / Superior, inferred from test scores)
- `english_test_score` (literal score string)
- `age` (integer)
- `regional_openness` (boolean)
- `years_work_experience` (float)
- `intended_state`
- `current_visa_status`

The extractor outputs strict JSON (no markdown fences, no prose) and is robust to common LLM formatting drift via regex-stripped parsing. Extracted attributes are merged into the profile with type coercion and overwrite semantics — later values replace earlier ones, so users can correct themselves ("actually my IELTS is 8.0").

### Routing classifier (`conversation_router.py`)

Before answering, every question is classified into `route: "answer" | "ask"`:

- **`answer`** — generic factual questions, or personalised questions where the profile has enough context
- **`ask`** — personalised questions where the profile is empty enough that an answer would be generic

The classifier returns JSON with the route, list of `missing_fields` (max 6), and a one-line reasoning string. Critical rules in the prompt:

- Strong bias toward answering
- Never ask twice — if the profile already has fields, the system has asked before
- Never ask for fields already populated
- Generic factual questions always answer

If the classifier returns malformed JSON or an unknown route, the system fails open to `"answer"` — a partial answer is better than a broken interaction.

### Follow-up generator (`conversation_router.py`)

When the route is `"ask"`, a separate LLM call generates the natural-language follow-up question. The prompt enforces:

- Warm, conversational tone (not corporate)
- One short response, under 5 lines
- Use the user's name if known
- Ask for missing fields as a short bulleted list
- Do not attempt to answer the question

The result: instead of a robotic "Please provide your field_of_study, age, english_level," the user gets *"Hey Aish — before I answer, could you tell me what you're studying, your age, and your English score?"*

---

## Generation

The grounded generator (`rag_pipeline.answer_question`):

1. Augments the query (conditional, see [Retrieval](#retrieval))
2. Runs hybrid retrieval to get the top-N chunks
3. Formats retrieved chunks into a numbered context block with `[Document N]`, title, source URL, file name, chunk ID
4. Builds the final prompt including the user profile, retrieved context, and the original question
5. Invokes the configured LLM with `temperature=0.0` for reproducibility
6. Returns a `RAGAnswer` dataclass with the answer text and the source documents

### Grounding prompt

The system prompt explicitly forbids answering from outside the retrieved context and requires the model to cite sources at the end of every answer. For calculation-style questions (points, fees, eligibility breakdowns), it forces an itemised list with explicit "unknown — depends on X" markers when the profile doesn't fully cover the calculation.

### Source citations

The UI renders sources in an expandable panel under each answer, showing:

- Document title
- Chunk ID
- Original source URL (clickable)
- A 300-character preview of the chunk content

This gives users a way to verify any claim against the original Home Affairs (or other government) page.

---

## Multi-provider LLM abstraction

A factory pattern (`app/llm_factory.py`) supports three providers behind a single `get_llm()` interface:

| Provider | Model | Notes |
|---|---|---|
| **Groq** (default) | `llama-3.3-70b-versatile` | Free, fast (~500ms-1s per call), high quality |
| **OpenAI** | `gpt-4o-mini` | Paid, highest consistency |
| **Google Gemini** | `gemini-2.0-flash` | Free tier with daily limits |

Provider selection is controlled by `LLM_PROVIDER` in `.env` and can be overridden per session via the sidebar settings. Each provider has its own API key environment variable (`GROQ_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`). Validation is lazy — only the provider being used needs its key.

Models are user-overridable via env vars (`GROQ_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`), so as providers deprecate or rename models the system doesn't require code changes.

Embeddings are deliberately not abstracted. OpenAI `text-embedding-3-small` is used for the production index (low cost: ~$0.001 for the full corpus build, ~$0.00002 per query). The codebase contains a documented swap path to local sentence-transformers for fully self-hosted deployments.

---

## Automated content refresh

Visa rules change frequently. To keep answers current, a **GitHub Actions workflow** runs on a weekly schedule:

1. Triggers via cron (every Monday at 06:00 UTC)
2. Spins up an Ubuntu runner with Python and Playwright
3. Runs `python app/scrape_visa_page.py` against every URL in `SCRAPE_URLS`
4. Diffs the new scraped content against the committed versions
5. If anything changed: runs `python app/build_index.py` to regenerate Chroma
6. Commits the updated raw files and index back to `main`
7. The Hugging Face Spaces deployment auto-redeploys from the new commit

This means a fee change on the Home Affairs site propagates to the live app within a week, with no manual intervention.

The workflow lives at `.github/workflows/refresh-content.yml`. Manual triggers are supported via `workflow_dispatch` for ad-hoc refreshes.

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| Frontend | Streamlit | Chat UI, sidebar profile, source panels, custom dark theme |
| LLM (default) | Groq — Llama 3.3 70B | Extraction, routing, follow-ups, grounded generation |
| LLM (alt) | OpenAI GPT-4o-mini, Gemini 2.0 Flash | Provider-switchable from UI |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim semantic vectors |
| Vector store | ChromaDB (persistent) | Dense retrieval, on-disk persistence |
| Lexical search | rank-bm25 via LangChain `BM25Retriever` | Keyword retrieval |
| Fusion | Reciprocal Rank Fusion (custom impl.) | Combine BM25 + dense rankings |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` | 800-char chunks, 150 overlap |
| Orchestration | LangChain | Document loaders, retrievers, embedding adapters |
| Scraping | Playwright (Chromium) | JS-rendered government pages with accordions |
| Automation | GitHub Actions | Weekly scrape + reindex + commit |
| Environment | python-dotenv | Local secret management |
| Language | Python 3.13 |

---

## Project structure

```
visamate/
├── .github/
│   └── workflows/
│       └── refresh-content.yml    # Weekly scrape + reindex
├── app/
│   ├── config.py                  # Single source of truth for all settings
│   ├── llm_factory.py             # Multi-provider LLM client factory
│   ├── ingest_local.py            # Load .txt files from data/raw/
│   ├── ingest_web.py              # Alt scraper using WebBaseLoader
│   ├── scrape_visa_page.py        # Playwright scraper with accordion expansion
│   ├── build_index.py             # Chunk + embed into Chroma
│   ├── hybrid_retriever.py        # BM25 + Chroma + RRF fusion
│   ├── rag_pipeline.py            # End-to-end retrieval → generation
│   ├── user_profile.py            # Profile dataclass + LLM extractor
│   ├── conversation_router.py     # Routing classifier + follow-up generator
│   ├── query.py                   # CLI chatbot
│   ├── debug_retrieval.py         # Inspect retrieval output for a query
│   └── debug_files.py             # Inspect corpus file sizes + chunk counts
├── data/
│   └── raw/                       # 16 curated + scraped source documents
├── docs/
│   └── screenshots/               # README screenshots
├── .streamlit/
│   └── config.toml                # Dark theme config
├── streamlit_app.py               # Main UI entry point
├── main.py                        # CLI entry hint
├── requirements.txt
├── .env.example
└── README.md
```

---

## Run it locally

### 1. Clone and install

```bash
git clone https://github.com/Aiswaryasudhir/visamate.git
cd visamate
python -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required: OpenAI key for embeddings (~$0.001 to build the entire index)
OPENAI_API_KEY=sk-...

# LLM provider — defaults to groq
LLM_PROVIDER=groq

# Groq key — free at https://console.groq.com
GROQ_API_KEY=gsk_...

# Optional: Google Gemini at https://aistudio.google.com/apikey
# GOOGLE_API_KEY=...
```

### 3. Build the index

```bash
cd app
python build_index.py
```

This chunks the source documents into ~185 chunks and embeds them into Chroma. One-time setup (~30 seconds, costs <$0.01 in embedding API).

### 4. Run the app

From the project root:

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### 5. (Optional) Re-scrape sources manually

GitHub Actions handles this on a schedule, but it can also be run locally:

```bash
playwright install chromium
cd app
python scrape_visa_page.py
python build_index.py
```

### 6. (Optional) Debug retrieval

To inspect what the retriever returns for a given question:

```bash
cd app
python debug_retrieval.py
```

Edit the `queries` list at the top of the file to test specific questions. Useful when answers seem off — usually the issue is retrieval, not generation.

---

## Configuration reference

All tunable parameters live in `app/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 800 | Character size of each retrieval chunk |
| `CHUNK_OVERLAP` | 150 | Characters shared between adjacent chunks |
| `VECTOR_K` | 6 | Candidates from dense (Chroma) retriever |
| `BM25_K` | 6 | Candidates from BM25 retriever |
| `FINAL_K` | 7 | Final documents after RRF fusion |
| `RRF_K_CONSTANT` | 60 | RRF k constant (Cormack et al., 2009) |
| `LLM_TEMPERATURE` | 0.0 | Generation temperature (low for grounding) |

Environment-overridable:

| Env var | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | One of: `groq`, `openai`, `gemini` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Google Gemini chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |

---

## Design decisions

**Chunk size = 800 / overlap = 150.** Tested against 300 and 1500. 800 is the sweet spot for visa content — large enough to capture full conditions (e.g., a complete 485 stream eligibility block) but small enough that retrieval stays focused.

**Title prepending to chunks.** Every chunk's content is prefixed with `[Document Title]`. Without this, BM25 retrieval competing on a keyword like "points" was dominated by large documents that mentioned points incidentally, while the dedicated points file lost. The title must be prepended *after* splitting — pre-split prepending only survives in chunk 0.

**Profile-aware query enrichment, conditionally applied.** Augmenting the retrieval query with profile keywords helps for generic questions ("what visa should I get") but hurts for topic-specific ones ("how many points"). A keyword gate in `_augment_query_with_profile` decides whether to augment. This was an empirical finding — initial implementation augmented unconditionally and broke the points retrieval.

**LLM-based extraction over regex.** User messages are parsed by an LLM into structured profile fields. Slower (~500ms) but robust to phrasing variation: "I'm 24", "my age is 24", "I just turned 24" all extract correctly. With Groq's latency the cost is negligible.

**Routing classifier, not heuristics.** Detecting whether a question needs context could be done with keyword triggers ("PR pathway", "points", etc.) but breaks on paraphrase. An LLM classifier with explicit rules generalises better and handles edge cases ("what should I do after my Master's?") without adding to a hardcoded list.

**Ask everything once.** The routing classifier is allowed to request up to 6 fields in a single follow-up. Earlier iterations limited it to 3 and progressively asked across turns — users found this frustrating. One consolidated ask, then answer.

**Session-scoped memory.** Profile lives in `st.session_state` only — no cross-session persistence, no accounts. Appropriate for a public assistant where most visits are single-session and adding auth would be infrastructure cost with no user-facing benefit.

**Fail open, not closed.** When extraction or routing returns malformed JSON, the system defaults to letting the question through to the generator. A partial or generic answer is always better than a broken interaction.

**Curated + scraped, not scraped-only.** Some content (fee tables, points breakdowns) scrapes into garbage because of JS-rendered widgets. Manually curating those as dated `.txt` files is more reliable than fighting the scraper.

---

## Limitations

- **Coverage is intentionally narrow.** Tourist visas, working holiday visas, partner visas, and citizenship are not in the corpus. The assistant will say "I can't verify that from the available sources" rather than hallucinate.
- **No real-time fee lookup.** Fees live in a curated `.txt` file dated to the last update. Always cross-check with the [Home Affairs Charge Estimator](https://immi.homeaffairs.gov.au/visas/getting-a-visa/fees-and-charges-for-visas) before any actual application.
- **No user accounts or persistence.** Each browser session starts fresh.
- **Single-language.** English only.
- **Stateless RAG, not full agent.** No tool use, no calculator, no web search at query time. The retrieval corpus is the entire world.

---

## Future scope

- [ ] **Retrieval evaluation suite** — annotated test set with hit-rate, MRR, and answer-quality metrics
- [ ] **Streaming responses** — token-by-token streaming for better perceived latency
- [ ] **Multi-turn conversation memory** beyond profile (full chat history in context)
- [ ] **Expanded coverage** — partner visa pathways, citizenship requirements
- [ ] **Reranker stage** — cross-encoder reranking on top of RRF for higher-precision retrieval

---

## Acknowledgements

Source content from the [Australian Department of Home Affairs](https://immi.homeaffairs.gov.au/), [Study Australia](https://www.studyaustralia.gov.au/), and individual state migration program pages. This project does not redistribute their content; it indexes and cites it.

---

## Contact

Built by **Aiswarya Sudhir**

[LinkedIn](https://www.linkedin.com/in/aiswarya-sudhir-008092178/) · [GitHub](https://github.com/Aiswaryasudhir) · [aiswaryasudhir@gmail.com](mailto:aiswaryasudhir@gmail.com)
