# utils/llm.py
"""
LLM configuration for CodeForge AI.

Uses Groq's GPT-OSS 120B model through CrewAI's native LLM interface.

Required Streamlit secret:
    GROQ_API_KEY = "gsk_..."

Optional:
    LLM_MODEL = "openai/gpt-oss-120b"
"""

import os
import streamlit as st
from crewai import LLM


# ---------------------------------------------------------------------------
# 1. API KEY
# ---------------------------------------------------------------------------

def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """
    Read API key from Streamlit secrets first,
    then fall back to environment variables.
    """

    # Streamlit Cloud secrets
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            value = st.secrets[key_name]

            if value and value != f"your_{key_name.lower()}_here":
                return str(value)

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
# 2. MODEL CONFIGURATION
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    """
    Get the model name from Streamlit secrets,
    then environment variables,
    then use the default model.
    """

    # Streamlit Cloud
    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            value = st.secrets["LLM_MODEL"]

            if value:
                return str(value)

    except Exception:
        pass

    # Environment variable
    value = os.getenv("LLM_MODEL")

    if value:
        return value

    return DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 3. LLM FACTORY
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.3) -> LLM:
    """
    Create a CrewAI LLM using Groq.

    The model is passed as:

        groq/openai/gpt-oss-120b

    CrewAI/LiteLLM handles the Groq provider internally.
    """

    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    # Make the API key available to libraries that read environment variables.
    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=8192,
    )


# ---------------------------------------------------------------------------
# 4. CONVENIENCE FUNCTIONS
# ---------------------------------------------------------------------------

def get_fast_llm(temperature: float = 0.3) -> LLM:
    """
    Lightweight tasks such as planning, review, and documentation.
    """
    return get_llm(temperature)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    """
    Heavy tasks such as coding, architecture, and debugging.
    """
    return get_llm(temperature)
