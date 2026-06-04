"""Decides whether the bot should ask for more context or answer directly.

Uses an LLM classifier to evaluate each question against the current
user profile. If the question is generic OR the needed profile is already
known, route to "answer". Otherwise, route to "ask" with a list of
specific things to ask for.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Literal

from langchain_core.language_models import BaseChatModel

from user_profile import UserProfile


Route = Literal["answer", "ask"]


@dataclass
class RoutingDecision:
    route: Route
    missing_fields: List[str]
    reasoning: str

    @classmethod
    def answer_directly(cls) -> "RoutingDecision":
        return cls(route="answer", missing_fields=[], reasoning="Generic question or sufficient context")


ROUTING_PROMPT = """You are a routing classifier for an Australian visa assistant.

Decide whether to ANSWER the user's question now OR ASK for context ONE TIME.

CRITICAL RULES:
1. Generic factual questions (visa fees, document lists, what is X) → ANSWER.
2. Personalised questions (PR pathways, points, eligibility, "which visa for me") →
   a. If the profile has AT LEAST 3 relevant fields populated → ANSWER with what you know,
      noting what other info would refine the answer.
   b. If the profile is nearly empty (0-2 fields) → ASK ONCE for ALL the fields you'd need
      to give a full personalised answer. Do not split this across multiple turns.
3. STRONG BIAS TOWARD ANSWERING. A partial answer with caveats is better than a follow-up.
4. NEVER ask for fields already in the profile.
5. NEVER ask twice. If profile already has fields, you have already asked. ANSWER now.

PROFILE FIELD NAMES (use these exact strings):
name,field_of_study, intended_occupation, english_level, english_test_score,
age, regional_openness, years_work_experience, intended_state, current_visa_status

Question:
\"\"\"{question}\"\"\"

Current user profile:
{profile}

Output ONLY valid JSON with this exact shape:
{{
  "route": "answer" OR "ask",
  "missing_fields": [list of field names, empty if route=answer],
  "reasoning": "one short sentence"
}}

JSON:"""


FOLLOW_UP_PROMPT = """You are VisaMate, a warm assistant for international students in Australia.

The user just asked something where you need a bit more about their situation before you
can give a useful answer.

Generate ONE short, friendly response that:
1. Acknowledges what they asked (warmly, conversationally).
2. Explains in ONE sentence why context matters here.
3. Asks for the specific missing pieces as a short bulleted list.
4. Does NOT attempt to answer yet.

Tone: like a friend who's been through this before. Warm, plain English, not corporate.
If you know their name, use it once at the start. Keep it under 5 lines.

User's question:
\"\"\"{question}\"\"\"

What you already know about them:
{profile}

Things you still need to know (ask in natural language):
{missing_fields}

Your response:"""


FIELD_LABELS = {
    "name": "your first name",
    "field_of_study": "your field of study or degree",
    "intended_occupation": "your intended occupation",
    "english_level": "your English proficiency level",
    "english_test_score": "your English test score (e.g. IELTS 7.5)",
    "age": "your current age",
    "regional_openness": "whether you're open to living in regional Australia",
    "years_work_experience": "how many years of work experience you have",
    "intended_state": "which Australian state you're targeting",
    "current_visa_status": "your current visa status (e.g. studying on 500, on 485)",
}


def route_question(
    question: str,
    profile: UserProfile,
    llm: BaseChatModel,
) -> RoutingDecision:
    """Decide whether to answer directly or ask for more context first."""
    prompt = ROUTING_PROMPT.format(
        question=question,
        profile=profile.to_prompt_block(),
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        route = data.get("route", "answer")
        if route not in ("answer", "ask"):
            route = "answer"

        missing = data.get("missing_fields", []) or []
        # Strip any fields already known
        known = set(profile.known_fields().keys())
        missing = [f for f in missing if f not in known]

        # Cap at 6
        missing = missing[:6]

        if route == "ask" and not missing:
            route = "answer"

        return RoutingDecision(
            route=route,
            missing_fields=missing,
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, AttributeError, ValueError):
        # Safe fallback: answer rather than ask, so we never block a generic question
        return RoutingDecision.answer_directly()


def generate_follow_up(
    question: str,
    profile: UserProfile,
    missing_fields: List[str],
    llm: BaseChatModel,
) -> str:
    """Generate a natural follow-up message asking for missing context."""
    missing_labels = "\n".join(
        f"- {FIELD_LABELS.get(f, f)}" for f in missing_fields
    )
    prompt = FOLLOW_UP_PROMPT.format(
        question=question,
        profile=profile.to_prompt_block(),
        missing_fields=missing_labels,
    )
    response = llm.invoke(prompt)
    return response.content.strip()