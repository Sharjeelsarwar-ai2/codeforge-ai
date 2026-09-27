# utils/llm.py
"""
LLM configuration for CodeForge AI using native CrewAI LLM.

Uses Groq's hosted models via LiteLLM / CrewAI.
"""

import os
import streamlit as st
from crewai import LLM


# ---------------------------------------------------------------------------
# 1. API Key helper
# ---------------------------------------------------------------------------
def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """
    Read an API key from Streamlit secrets first,
    then fall back to environment variables.
    """
    # Try Streamlit Cloud secrets
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            value = st.secrets[key_name]
            if value and value != f"your_{key_name.lower()}_here":
                return str(value).strip()
    except Exception:
        pass

    # Environment variable fallback
    value = os.getenv(key_name)
    if value:
        return str(value).strip()

    raise ValueError(
        f"{key_name} not found. "
        f"Add it to Streamlit Secrets (.streamlit/secrets.toml) or set it as an environment variable."
    )


# ---------------------------------------------------------------------------
# 2. Model configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    """
    Get the model name from Streamlit secrets, with a sensible default.
    Ensures the 'groq/' prefix is included for LiteLLM routing.
    """
    model = DEFAULT_MODEL
    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            model = str(st.secrets["LLM_MODEL"]).strip()
    except Exception:
        pass

    model = os.getenv("LLM_MODEL", model)

    # CrewAI / LiteLLM requires the provider prefix: 'groq/<model_name>'
    if not model.startswith("groq/"):
        return f"groq/{model}"
    return model


# ---------------------------------------------------------------------------
# 3. LLM factory using native CrewAI LLM
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.3) -> LLM:
    """
    Create and return a native CrewAI LLM instance.

    Args:
        temperature: 0.0 – 1.0 (lower = more deterministic)
    """
    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    # Set in OS environment so LiteLLM and CrewAI can access it internally
    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# 4. Convenience shortcuts
# ---------------------------------------------------------------------------
def get_fast_llm(temperature: float = 0.2) -> LLM:
    """Strict / consistent tasks (review, qa, docs)."""
    return get_llm(temperature=temperature)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    """Coding and planning tasks."""
    return get_llm(temperature=temperature)
