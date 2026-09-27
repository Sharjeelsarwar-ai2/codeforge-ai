# utils/llm.py
"""
LLM configuration for CodeForge AI.

Uses Groq's hosted GPT-OSS 120B model for all agents.

Streamlit secrets required:
    GROQ_API_KEY = "gsk_..."

Optional overrides:
    LLM_MODEL = "openai/gpt-oss-120b"
"""

import os
import streamlit as st
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# 1. API Key helper
# ---------------------------------------------------------------------------
def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """
    Read an API key from Streamlit secrets first,
    then fall back to environment variables.
    """
    # Streamlit Cloud secrets
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            value = st.secrets[key_name]
            if value and value != f"your_{key_name.lower()}_here":
                return value
    except Exception:
        pass

    # Environment variable fallback
    value = os.getenv(key_name)
    if value:
        return value

    raise ValueError(
        f"{key_name} not found. "
        f"Add it to Streamlit Secrets or set it as an environment variable."
    )


# ---------------------------------------------------------------------------
# 2. Model configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    """
    Get the model name from Streamlit secrets, with a sensible default.
    """
    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            return st.secrets["LLM_MODEL"]
    except Exception:
        pass

    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# 3. LLM factory
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.3) -> ChatGroq:
    """
    Create and return a ChatGroq LLM instance running GPT-OSS 120B.

    Args:
        temperature: 0.0 – 1.0  (lower = more deterministic)
    """
    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    # Also inject into env so CrewAI internals can find it
    os.environ["GROQ_API_KEY"] = api_key

    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=8192,
    )


# ---------------------------------------------------------------------------
# 4. Convenience shortcuts (kept for backwards compatibility)
# ---------------------------------------------------------------------------
def get_fast_llm(temperature: float = 0.3) -> ChatGroq:
    """Lightweight tasks (planning, review, docs) — same 120B model."""
    return get_llm(temperature)


def get_powerful_llm(temperature: float = 0.3) -> ChatGroq:
    """Heavy tasks (coding, design, debugging) — same 120B model."""
    return get_llm(temperature)