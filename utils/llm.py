# utils/llm.py
"""
LLM configuration for CodeForge AI using native CrewAI LLM + LiteLLM.

Includes a custom callback that strips 'cache_breakpoint' and 'cache_control'
from all messages before they reach Groq's API (which does not support them).
"""

import os
import streamlit as st
from crewai import LLM
import litellm


# ---------------------------------------------------------------------------
# CRITICAL: Custom callback to strip unsupported cache properties
# ---------------------------------------------------------------------------
def _strip_cache_properties(kwargs: dict) -> dict:
    """
    Remove cache_breakpoint, cache_control, and any other unsupported
    properties from all messages before sending to Groq.
    """
    messages = kwargs.get("messages", [])
    if not messages:
        return kwargs

    cleaned_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Remove problematic keys
            cleaned_msg = {
                k: v for k, v in msg.items()
                if k not in ("cache_breakpoint", "cache_control", "providerOptions")
            }

            # Also clean nested content if it's a list of parts
            if isinstance(cleaned_msg.get("content"), list):
                cleaned_content = []
                for part in cleaned_msg["content"]:
                    if isinstance(part, dict):
                        cleaned_part = {
                            k: v for k, v in part.items()
                            if k not in ("cache_breakpoint", "cache_control")
                        }
                        cleaned_content.append(cleaned_part)
                    else:
                        cleaned_content.append(part)
                cleaned_msg["content"] = cleaned_content

            cleaned_messages.append(cleaned_msg)
        else:
            cleaned_messages.append(msg)

    kwargs["messages"] = cleaned_messages
    return kwargs


# Monkey-patch litellm.completion to always clean messages
_original_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    """Wrapper that strips cache properties before every LiteLLM completion call."""
    kwargs = _strip_cache_properties(kwargs)
    # Also remove any top-level unsupported params
    kwargs.pop("cache_breakpoint", None)
    kwargs.pop("cache_control", None)
    return _original_completion(*args, **kwargs)


litellm.completion = _patched_completion

# Also patch acompletion (async version)
_original_acompletion = litellm.acompletion


async def _patched_acompletion(*args, **kwargs):
    kwargs = _strip_cache_properties(kwargs)
    kwargs.pop("cache_breakpoint", None)
    kwargs.pop("cache_control", None)
    return await _original_acompletion(*args, **kwargs)


litellm.acompletion = _patched_acompletion

# Tell LiteLLM to silently drop any unsupported parameters
litellm.drop_params = True


# ---------------------------------------------------------------------------
# API Key helper
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.3) -> LLM:
    """Create and return a native CrewAI LLM instance configured for Groq."""
    api_key = get_api_key("GROQ_API_KEY")
    model_name = get_model_name()

    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        cache=False,
    )


def get_fast_llm(temperature: float = 0.2) -> LLM:
    return get_llm(temperature=temperature)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    return get_llm(temperature=temperature)
