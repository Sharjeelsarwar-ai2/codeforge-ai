# utils/llm.py
"""LLM configuration for CodeForge AI.

The application runs ten CrewAI agents sequentially through Groq. Groq can
return HTTP 429 responses for short-lived RPM/TPM bursts, so this module keeps
requests paced and retries only transient rate-limit failures. Permanent quota
errors are surfaced immediately instead of being retried pointlessly.

Required Streamlit secret or environment variable:
    GROQ_API_KEY = "gsk_..."

Optional configuration:
    LLM_MODEL = "openai/gpt-oss-120b"
    GROQ_MIN_REQUEST_INTERVAL_SECONDS = "1.0"
    GROQ_MAX_RETRIES = "3"
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from typing import Any

import streamlit as st
from crewai import LLM


# -------------------------------------------------------------------
# API KEY
# -------------------------------------------------------------------

def get_api_key(key_name: str = "GROQ_API_KEY") -> str:
    """Get a provider API key from Streamlit Secrets or the environment."""
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            value = st.secrets[key_name]
            if value and value != f"your_{key_name.lower()}_here":
                return str(value)
    except Exception:
        # Streamlit Secrets raises when the app is run outside Streamlit.
        pass

    value = os.getenv(key_name)
    if value:
        return value

    raise ValueError(
        f"{key_name} not found. Add it to Streamlit Secrets or set it as an environment variable."
    )


# -------------------------------------------------------------------
# MODEL AND RATE-LIMIT SETTINGS
# -------------------------------------------------------------------

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-120b"
_DEFAULT_MIN_REQUEST_INTERVAL = 1.0
_DEFAULT_MAX_RETRIES = 3


def get_model_name() -> str:
    """Read the configured model name and remove an optional provider prefix."""
    try:
        if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
            value = st.secrets["LLM_MODEL"]
            if value:
                configured = str(value).strip()
                if configured.removeprefix("groq/") == "llama-3.1-8b-instant":
                    return DEFAULT_MODEL
                return configured
    except Exception:
        pass

    configured = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if configured.removeprefix("groq/") == "llama-3.1-8b-instant":
        return DEFAULT_MODEL
    return configured


def get_fallback_model_name() -> str:
    """Read the lower-cost model used when the primary model hits TPM limits."""
    try:
        if hasattr(st, "secrets") and "FALLBACK_LLM_MODEL" in st.secrets:
            value = st.secrets["FALLBACK_LLM_MODEL"]
            if value:
                configured = str(value).strip()
                if configured.removeprefix("groq/") == "llama-3.1-8b-instant":
                    return DEFAULT_FALLBACK_MODEL
                return configured
    except Exception:
        pass
    configured = os.getenv("FALLBACK_LLM_MODEL", DEFAULT_FALLBACK_MODEL).strip() or DEFAULT_FALLBACK_MODEL
    if configured.removeprefix("groq/") == "llama-3.1-8b-instant":
        return DEFAULT_FALLBACK_MODEL
    return configured


def _groq_model(model_name: str) -> str:
    """Return exactly one LiteLLM Groq provider prefix."""
    if model_name.startswith("groq/"):
        return model_name
    return f"groq/{model_name}"


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# A process-wide gate is important because each CrewAI agent receives its own
# LLM object, but all of them share the same Groq key and rate limit.
_request_gate = threading.Lock()
_last_request_at = 0.0


def _pace_requests() -> None:
    """Keep sequential agent calls from creating an RPM burst."""
    global _last_request_at
    interval = _env_float(
        "GROQ_MIN_REQUEST_INTERVAL_SECONDS", _DEFAULT_MIN_REQUEST_INTERVAL
    )
    if interval <= 0:
        return

    with _request_gate:
        wait_for = interval - (time.monotonic() - _last_request_at)
        if wait_for > 0:
            time.sleep(wait_for)
        _last_request_at = time.monotonic()


def _is_retryable_rate_limit(error: BaseException) -> bool:
    """Recognize transient 429/rate-limit errors across LiteLLM versions."""
    text = str(error).lower()
    error_name = type(error).__name__.lower()

    if any(marker in text for marker in ("insufficient_quota", "quota exceeded", "per day")):
        return False

    return (
        "ratelimit" in error_name
        or "too many requests" in text
        or "rate limit" in text
        or "rate_limit_exceeded" in text
        or "status code: 429" in text
        or "status_code=429" in text
        or "http 429" in text
        or re.search(r"\b429\b", text) is not None
    )


def _retry_delay(error: BaseException, attempt: int) -> float:
    """Use a provider-suggested delay when present, otherwise exponential backoff."""
    text = str(error)
    patterns = (
        r"retry[- ]after[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        r"try again in[\s:]+([0-9]+(?:\.[0-9]+)?)\s*s?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return min(60.0, max(0.5, float(match.group(1)))) + random.uniform(0, 0.25)

    # Jitter prevents multiple Streamlit workers from retrying simultaneously.
    return min(60.0, 2.0 ** attempt) + random.uniform(0, 0.25)


class RateLimitAwareLLM(LLM):
    """CrewAI LLM with pacing and bounded retries for transient Groq 429s."""

    def call(self, messages: list[dict[str, str]], callbacks: list[Any] | None = None) -> str:
        retries = _env_int("GROQ_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        for attempt in range(retries + 1):
            _pace_requests()
            try:
                return super().call(messages, callbacks or [])
            except Exception as error:
                if not _is_retryable_rate_limit(error) or attempt >= retries:
                    raise
                # Groq TPM errors include the prompt plus the requested output
                # budget. Shrink the next request so the retry can fit after
                # the provider-suggested cooldown instead of repeating the same
                # oversized request.
                if self.max_tokens:
                    self.max_tokens = max(256, self.max_tokens // 2)

                # GPT-OSS 120B commonly has a very small free-tier TPM window.
                # Switch once to the lightweight Groq model so a hard limit on
                # the primary model does not abort the complete pipeline.
                fallback_model = _groq_model(get_fallback_model_name())
                switched_to_fallback = (
                    not getattr(self, "_fallback_used", False)
                    and self.model != fallback_model
                )
                if switched_to_fallback:
                    self.model = fallback_model
                    self._fallback_used = True
                    self.max_tokens = min(self.max_tokens or 1024, 1024)

                delay = _retry_delay(error, attempt)
                # A model switch uses a separate model quota; do not make the
                # user wait through the old model's cooldown unnecessarily.
                time.sleep(min(delay, 1.0) if switched_to_fallback else delay)

        raise RuntimeError("LLM call exhausted its retry budget")


# -------------------------------------------------------------------
# COMMON LLM FACTORY
# -------------------------------------------------------------------

def _create_llm(temperature: float = 0.3, max_tokens: int = 4096) -> LLM:
    """Create a paced, retrying CrewAI LLM with a bounded output budget."""
    api_key = get_api_key("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = api_key

    return RateLimitAwareLLM(
        model=_groq_model(get_model_name()),
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------------------------------------------------
# PUBLIC LLM FACTORIES
# -------------------------------------------------------------------

def get_llm(temperature: float = 0.3, max_tokens: int = 4096) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=max_tokens)


def get_planning_llm(temperature: float = 0.3) -> LLM:
    # Planning output is structured and should not consume the whole TPM window.
    return _create_llm(temperature=temperature, max_tokens=768)


def get_development_llm(temperature: float = 0.3) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=2048)


def get_review_llm(temperature: float = 0.2) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=1024)


def get_debug_llm(temperature: float = 0.2) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=2048)


def get_fast_llm(temperature: float = 0.3) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=1024)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=2048)
