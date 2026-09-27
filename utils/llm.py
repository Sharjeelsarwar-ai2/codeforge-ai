# utils/llm.py
"""
LLM configuration for CodeForge AI using native CrewAI LLM + LiteLLM.
Configured to prevent prompt caching issues with Groq.
"""

import os
import streamlit as st
from crewai import LLM
import litellm

# --- CRITICAL FIXES FOR GROQ API COMPATIBILITY ---
# 1. Disable CrewAI's prompt caching which injects 'cache_breakpoint'
os.environ["CREWAI_DISABLE_PROMPT_CACHING"] = "true"

# 2. Tell LiteLLM to drop any parameters not natively supported by Groq
litellm.drop_params = True


def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """Read API key from Streamlit secrets first, then environment variables."""
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            val = st.secrets[key_name]
            if val and val != f"your_{key_name.lower()}_here":
                return str(val).strip()
    except Exception:
        pass

    val = os.getenv(key_name)
    if val:
        return str(val).strip()

    raise ValueError(
        f"{key_name} not found. Add it to Streamlit Secrets (.streamlit/secrets.toml)."
    )


DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_model_name() -> str:
    """Get the model name and ensure 'groq/' prefix for LiteLLM."""
    model = DEFAULT_MODEL
    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            model = str(st.secrets["LLM_MODEL"]).strip()
    except Exception:
        pass

    model = os.getenv("LLM_MODEL", model)

    if not model.startswith("groq/"):
        return f"groq/{model}"
    return model


def get_llm(temperature: float = 0.3) -> LLM:
    """Create and return a native CrewAI LLM instance configured for Groq."""
    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    # LiteLLM reads GROQ_API_KEY directly from the environment
    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        # Disable CrewAI's internal agent-level caching
        cache=False,
    )


def get_fast_llm(temperature: float = 0.2) -> LLM:
    return get_llm(temperature=temperature)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    return get_llm(temperature=temperature)
