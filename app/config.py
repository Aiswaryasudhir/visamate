"""Centralized configuration for the Visa Navigator project."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths (absolute, so scripts work from any directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
COLLECTION_NAME = "visa_docs"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
VECTOR_K = 6
BM25_K = 6
FINAL_K = 7
RRF_K_CONSTANT = 60

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Active provider — choose 'openai' | 'groq' | 'gemini'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_TEMPERATURE = 0.0

# Per-provider model names (overridable via env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Embeddings stay on OpenAI (cheap + high quality)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---------------------------------------------------------------------------
# Sources — local file -> origin URL mapping (used as citation metadata)
# ---------------------------------------------------------------------------
SOURCE_MAP: dict[str, str] = {
    # --- Existing: Student 500 / dependents / work ---
    "student_500_main.txt": "https://immi.homeaffairs.gov.au/visas/already-have-a-visa/check-visa-details-and-conditions/see-your-visa-conditions?product=500",
    "subclass_500_details.txt": "https://immi.homeaffairs.gov.au/visas/already-have-a-visa/check-visa-details-and-conditions/see-your-visa-conditions?product=500",
    "work_conditions.txt": "https://www.education.gov.au/international-education/support-international-students/rights-international-students-work",
    "dependent_visa.txt": "https://checkworkrights.com.au/resources/student-visa-subclass-500-secondary-applicant/",
    "dependent_visa_2.txt": "https://www.studyaustralia.gov.au/en/plan-your-move/bringing-your-family",
    "after_visa_expiry.txt": "https://oshcaustralia.com.au/en/blog/staying-in-australia-after-my-student-course-ends",

    # --- Existing: GS + 485 ---
    "genuine_student_requirement.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/genuine-student-requirement",
    "temporary_graduate_485_overview.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485",
    "tg_post_higher_education_work.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-higher-education-work",
    "tg_post_vocational_education_work.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-vocational-education-work",
    "tg_485_changes.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/changes",

    # --- Existing: Curated reference ---
    "visa_costs_2026.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/fees-and-charges-for-visas",
    "documents_required.txt": "https://immi.homeaffairs.gov.au/visas/web-evidentiary-tool",

    # --- NEW: Skilled migration / PR pathway pages ---
    "skilled_independent_189.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189",
    "skilled_nominated_190.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-nominated-190",
    "skilled_work_regional_491.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-work-regional-provisional-491",
    "employer_nomination_186.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/employer-nomination-scheme-186",
    "skills_in_demand_482.txt": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skills-in-demand-482",

    # --- NEW: Curated PR-pathway reference ---
    "pr_pathways_overview.txt": "https://immi.homeaffairs.gov.au/visas/working-in-australia/skillselect",
    "points_test_explainer.txt": "https://immi.homeaffairs.gov.au/visas/working-in-australia/skillselect/points-table",
    "state_nomination_overview.txt": "https://immi.homeaffairs.gov.au/visas/working-in-australia/skillselect/state-and-territory-nominated-visa-pathways",
}

SCRAPE_URLS: dict[str, str] = {
    # --- Existing ---
    "student_500_main": "https://immi.homeaffairs.gov.au/visas/already-have-a-visa/check-visa-details-and-conditions/see-your-visa-conditions?product=500",
    "work_conditions": "https://www.education.gov.au/international-education/support-international-students/rights-international-students-work",
    "dependent_visa": "https://checkworkrights.com.au/resources/student-visa-subclass-500-secondary-applicant/",
    "dependent_visa_2": "https://www.studyaustralia.gov.au/en/plan-your-move/bringing-your-family",
    "after_visa_expiry": "https://oshcaustralia.com.au/en/blog/staying-in-australia-after-my-student-course-ends",
    "genuine_student_requirement": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/genuine-student-requirement",
    "temporary_graduate_485_overview": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485",
    "tg_post_higher_education_work": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-higher-education-work",
    "tg_post_vocational_education_work": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/post-vocational-education-work",
    "tg_485_changes": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-graduate-485/changes",

    # --- NEW: PR / skilled migration pages ---
    "skilled_independent_189": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-independent-189",
    "skilled_nominated_190": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-nominated-190",
    "skilled_work_regional_491": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skilled-work-regional-provisional-491",
    "employer_nomination_186": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/employer-nomination-scheme-186",
    "skills_in_demand_482": "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/skills-in-demand-482",
}


def validate_environment() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not found. Create a .env file in the project root "
            "with: OPENAI_API_KEY=sk-... (see .env.example)"
        )