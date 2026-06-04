"""User profile data model + LLM-powered extraction from free-text messages.

The profile is built up incrementally over a conversation. Each user message
is checked for any new attributes; the bot only ever asks for what's missing
and relevant to the current question.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from langchain_core.language_models import BaseChatModel


@dataclass
class UserProfile:
    """All the attributes we might collect from a user across a conversation.

    Every field is optional. None means "not yet known". The bot decides what
    to ask for based on the current question, not based on which fields are
    empty.
    """
    name: Optional[str] = None                   # user's first name (used in conversation)
    field_of_study: Optional[str] = None        # e.g. "Master of Data Science"
    intended_occupation: Optional[str] = None    # e.g. "Data Scientist", ANZSCO if known
    english_level: Optional[str] = None          # "Competent" | "Proficient" | "Superior"
    english_test_score: Optional[str] = None     # e.g. "IELTS 7.5"
    age: Optional[int] = None                    # current age
    regional_openness: Optional[bool] = None     # willing to live in regional Australia
    years_work_experience: Optional[float] = None
    intended_state: Optional[str] = None         # "Victoria", "NSW", etc.
    current_visa_status: Optional[str] = None    # "studying on 500", "on 485", "offshore", etc.

    def is_empty(self) -> bool:
        return all(v is None for v in asdict(self).values())

    def known_fields(self) -> dict:
        """Return only the fields that have values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_prompt_block(self) -> str:
        """Render as a compact block for inclusion in LLM prompts."""
        known = self.known_fields()
        if not known:
            return "(no profile information yet)"

        lines = []
        labels = {
            "name": "Name",
            "field_of_study": "Field of study",
            "intended_occupation": "Intended occupation",
            "english_level": "English level",
            "english_test_score": "English test score",
            "age": "Age",
            "regional_openness": "Open to regional Australia",
            "years_work_experience": "Years of work experience",
            "intended_state": "Intended state",
            "current_visa_status": "Current visa status",
        }
        for key, value in known.items():
            label = labels.get(key, key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    def to_sidebar_dict(self) -> dict:
        """Friendly key->value pairs for sidebar display."""
        known = self.known_fields()
        labels = {
            "name": "👋 Name",
            "field_of_study": "🎓 Field of study",
            "intended_occupation": "💼 Occupation",
            "english_level": "🗣️ English level",
            "english_test_score": "📝 Test score",
            "age": "🎂 Age",
            "regional_openness": "🏞️ Regional OK",
            "years_work_experience": "📅 Work exp (yrs)",
            "intended_state": "📍 State",
            "current_visa_status": "🛂 Current visa",
        }
        out = {}
        for key, value in known.items():
            label = labels.get(key, key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            out[label] = str(value)
        return out


EXTRACTION_PROMPT = """You extract user attributes from a chat message.

The user is an international student or graduate in Australia asking about visas.
Extract ONLY attributes the user explicitly states or strongly implies in THIS message.
Do NOT guess. Do NOT carry over from prior context (that is handled separately).
If an attribute is not in this message, OMIT it from the JSON entirely.

Available attributes (JSON keys):
- name: the user's first name if they share it (e.g. "I'm Aish" → "Aish")
- field_of_study: course or degree they are doing (e.g. "Master of Data Science")
- intended_occupation: job they want or have (e.g. "Data Scientist", "Software Engineer")
- english_level: one of "Competent", "Proficient", "Superior" — INFER from test scores:
    Competent ~ IELTS 6.0 in each band / PTE 50
    Proficient ~ IELTS 7.0 in each band / PTE 65
    Superior ~ IELTS 8.0 in each band / PTE 79
- english_test_score: literal score they mentioned (e.g. "IELTS 7.5 overall", "PTE 75")
- age: integer age in years
- regional_openness: true if they say they're open to regional / willing to move outside major cities;
    false if they say they want Sydney/Melbourne/Brisbane only
- years_work_experience: number (can be decimal)
- intended_state: Australian state or territory (e.g. "Victoria", "NSW")
- current_visa_status: short description (e.g. "studying on 500", "on 485", "offshore applicant")

Output ONLY a valid JSON object. No prose, no markdown, no code fences.
If the message contains NO extractable attributes, output: {{}}

Message:
\"\"\"{message}\"\"\"

JSON:"""


def extract_attributes(message: str, llm: BaseChatModel) -> dict:
    """Use the LLM to extract any user attributes mentioned in a message.

    Returns a dict of {field_name: value}. Empty dict if nothing found.
    """
    prompt = EXTRACTION_PROMPT.format(message=message)
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip code fences if the model added them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {}


def merge_into_profile(profile: UserProfile, new_attrs: dict) -> List[str]:
    """Merge extracted attributes into the profile, returning a list of
    fields that were updated. Later values overwrite earlier ones (so the
    user can correct themselves)."""
    updated = []
    valid_fields = set(asdict(profile).keys())

    for key, value in new_attrs.items():
        if key not in valid_fields:
            continue
        if value is None or value == "":
            continue

        # Light type coercion
        if key == "age":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        elif key == "years_work_experience":
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        elif key == "regional_openness":
            if isinstance(value, str):
                value = value.lower() in {"true", "yes", "y", "1"}

        current = getattr(profile, key)
        if current != value:
            setattr(profile, key, value)
            updated.append(key)

    return updated