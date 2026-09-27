# utils/llm.py
"""
LLM configuration for CodeForge AI.

Uses Groq's GPT-OSS 120B model through CrewAI's native LLM interface.

Different token budgets are used for different phases:

- Planning agents: smaller responses to stay within Groq TPM limits
- Development agents: larger responses because they generate code
- Review/QA agents: medium responses
- Debug agents: larger responses when code fixes are required

Required Streamlit secret:
    GROQ_API_KEY = "gsk_..."

Optional:
    LLM_MODEL = "openai/gpt-oss-120b"
"""

import os
import streamlit as st
from crewai import LLM


# -------------------------------------------------------------------
# API KEY
# -------------------------------------------------------------------

def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """
    Get the Groq API key from Streamlit Secrets or environment variables.
    """

    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            value = st.secrets[key_name]

            if value and value != f"your_{key_name.lower()}_here":
                return str(value)

    except Exception:
        pass

    value = os.getenv(key_name)

    if value:
        return value

    raise ValueError(
        f"{key_name} not found. "
        f"Add it to Streamlit Secrets or set it as an environment variable."
    )


# -------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    """
    Get model name from Streamlit Secrets or environment variables.
    """

    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            value = st.secrets["LLM_MODEL"]

            if value:
                return str(value)

    except Exception:
        pass

    value = os.getenv("LLM_MODEL")

    if value:
        return value

    return DEFAULT_MODEL


# -------------------------------------------------------------------
# COMMON LLM FACTORY
# -------------------------------------------------------------------

def _create_llm(
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLM:
    """
    Create a CrewAI LLM instance.

    max_tokens is configurable so different CodeForge phases
    can use different response budgets.
    """

    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    # Make sure LiteLLM/Groq can find the key.
    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=f"groq/{model_name}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------------------------------------------------
# GENERAL LLM
# -------------------------------------------------------------------

def get_llm(
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLM:
    """
    General-purpose LLM.

    Default is intentionally lower than the previous 8192-token
    configuration to reduce Groq TPM usage.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------------------------------------------------
# PLANNING LLM
# -------------------------------------------------------------------

def get_planning_llm(
    temperature: float = 0.3,
) -> LLM:
    """
    LLM specifically for planning agents.

    Planning should produce concise specifications rather than
    enormous responses.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=3000,
    )


# -------------------------------------------------------------------
# DEVELOPMENT LLM
# -------------------------------------------------------------------

def get_development_llm(
    temperature: float = 0.3,
) -> LLM:
    """
    LLM for actual code generation.

    Development needs more output space than planning.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=8192,
    )


# -------------------------------------------------------------------
# REVIEW / QA LLM
# -------------------------------------------------------------------

def get_review_llm(
    temperature: float = 0.2,
) -> LLM:
    """
    LLM for code review and QA reports.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=4000,
    )


# -------------------------------------------------------------------
# DEBUG LLM
# -------------------------------------------------------------------

def get_debug_llm(
    temperature: float = 0.2,
) -> LLM:
    """
    LLM for debugging and fixing generated code.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=6000,
    )


# -------------------------------------------------------------------
# FAST / POWERFUL COMPATIBILITY FUNCTIONS
# -------------------------------------------------------------------

def get_fast_llm(
    temperature: float = 0.3,
) -> LLM:
    """
    Compatibility helper for lightweight tasks.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=3000,
    )


def get_powerful_llm(
    temperature: float = 0.3,
) -> LLM:
    """
    Compatibility helper for larger tasks.
    """

    return _create_llm(
        temperature=temperature,
        max_tokens=8192,
    )
