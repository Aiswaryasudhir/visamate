"""End-to-end RAG pipeline with optional user profile injection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from hybrid_retriever import hybrid_retrieve
from llm_factory import get_llm
from user_profile import UserProfile

GROUNDED_PROMPT = """You are VisaMate, a warm and helpful assistant for international students
navigating Australian visas. You're knowledgeable but speak like a friendly human, not a textbook.

TONE:
- Warm but professional. Like a slightly older friend who happens to know visa stuff.
- Use the user's name naturally when you know it, but don't overdo it (1-2 times max per response).
- Avoid corporate phrases like "I would be happy to assist you" or "based on the provided context".
- Keep things concrete. Use plain English, short sentences, real numbers.

GROUNDING:
- Answer using ONLY the retrieved context below.
- If the answer isn't in the context, say: "I can't verify that from the sources I have."

{profile_section}

ANSWER QUALITY RULES:
1. For calculations or breakdowns (points, fees, eligibility checks), produce a COMPLETE
   itemised list. Go through EVERY relevant category in the source and either:
     (a) assign a value using the user profile, OR
     (b) mark it "unknown — depends on X" if the profile doesn't cover it.
   End with a TOTAL.

2. Read tables carefully. If a table maps ranges to values (e.g. "18-24: 25 points,
   25-32: 30 points"), match the user's exact value to the correct row.

3. Tailor the answer to the user's profile. Don't give a generic "it depends" if their
   profile already answers the question.

4. Be honest about uncertainty. If something genuinely depends on info you don't have
   (e.g. skills assessment outcome, occupation list status), say so briefly — but
   still give the partial answer.

5. End with a short Sources list, formatted as:
   [Source: <title> — <url>]

Context:
{context}

Question:
{question}

Answer:"""


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: List[Document]


def _profile_section(profile: Optional[UserProfile]) -> str:
    if profile is None or profile.is_empty():
        return "User profile: (not provided)"
    return (
        "User profile (use this to personalize your answer):\n"
        f"{profile.to_prompt_block()}"
    )


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "Unknown")
        file_name = doc.metadata.get("file_name", "Unknown file")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        parts.append(
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"File: {file_name}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def build_prompt(
    question: str,
    docs: List[Document],
    profile: Optional[UserProfile] = None,
) -> str:
    return GROUNDED_PROMPT.format(
        profile_section=_profile_section(profile),
        context=format_context(docs),
        question=question,
    )


def _augment_query_with_profile(question: str, profile: Optional[UserProfile]) -> str:
    """Decide whether to enrich the retrieval query with profile keywords.

    We only augment for GENERIC questions where the user hasn't named a topic.
    For topic-specific questions (points, fees, documents, GS, specific visa
    subclasses), the raw question already retrieves better than an augmented one.
    """
    if profile is None or profile.is_empty():
        return question

    # If the question itself contains a strong topic anchor, do not augment.
    # Augmentation drowns out the topic signal in hybrid retrieval.
    topic_keywords = [
        "points", "point", "fee", "fees", "cost", "charge",
        "document", "documents", "checklist",
        "genuine student", "gs ",
        "189", "190", "491", "482", "485", "186", "500",
        "english", "ielts", "pte", "toefl",
        "oshc", "health cover",
    ]
    lowered = question.lower()
    if any(kw in lowered for kw in topic_keywords):
        return question

    # Generic question — enrich with profile to bias retrieval toward
    # occupation/state/regional content the user cares about.
    extras = []
    if profile.field_of_study:
        extras.append(profile.field_of_study)
    if profile.intended_occupation:
        extras.append(profile.intended_occupation)
    if profile.intended_state:
        extras.append(profile.intended_state)
    if profile.regional_openness is True:
        extras.append("regional")
    if not extras:
        return question
    return f"{question} {' '.join(extras)}"


def answer_question(
    question: str,
    llm: Optional[BaseChatModel] = None,
    profile: Optional[UserProfile] = None,
) -> RAGAnswer:
    if llm is None:
        llm = get_llm()

    retrieval_query = _augment_query_with_profile(question, profile)
    docs = hybrid_retrieve(retrieval_query)
    prompt = build_prompt(question, docs, profile=profile)
    response = llm.invoke(prompt)

    return RAGAnswer(
        question=question,
        answer=response.content,
        sources=docs,
    )


__all__ = ["RAGAnswer", "answer_question", "build_prompt", "format_context", "get_llm"]