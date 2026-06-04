"""Factory for constructing LLM clients across providers."""
from __future__ import annotations

import os
from typing import Literal, Optional

from langchain_core.language_models import BaseChatModel

from config import (
    GEMINI_MODEL,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_MODEL,
)

Provider = Literal["openai", "groq", "gemini"]

PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

PROVIDER_MODEL = {
    "openai": OPENAI_MODEL,
    "groq": GROQ_MODEL,
    "gemini": GEMINI_MODEL,
}


def _validate_key(provider: str) -> None:
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Use one of: {list(PROVIDER_KEY_ENV)}."
        )
    required = PROVIDER_KEY_ENV[provider]
    if not os.getenv(required):
        raise RuntimeError(
            f"{required} not found in environment. "
            f"Add it to your .env file (see .env.example)."
        )


def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Return a configured LLM client for the given provider.

    If provider is None, uses LLM_PROVIDER from config (default 'groq').
    """
    provider = (provider or LLM_PROVIDER).lower()
    _validate_key(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=LLM_TEMPERATURE)

    raise ValueError(f"Unhandled provider '{provider}'")