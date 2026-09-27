# utils/llm.py
"""LLM configuration for CodeForge AI.

The application runs ten CrewAI agents sequentially through Groq. Groq can
return HTTP 429 responses for short-lived RPM/TPM bursts, so this module keeps
requests paced and retries only transient rate-limit failures. Permanent quota
errors are surfaced immediately instead of being retried pointlessly.

Required Streamlit secret or environment variable:
    GROQ_API_KEY = "gsk_..."

Optional configuration:
    LLM_MODEL = "qwen/qwen3.8-27b"
    DEVELOPMENT_LLM_MODEL = "qwen/qwen3.8-27b"
    GROQ_MIN_REQUEST_INTERVAL_SECONDS = "1.0"
    GROQ_MAX_RETRIES = "3"
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from typing import Any

import streamlit as st
from crewai import LLM


logger = logging.getLogger(__name__)


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

# Free, open-source, cloud-hosted model used for code generation.
# Qwen3 32B is available on Groq's free tier and is specifically strong at
# code generation (65.7% on LiveCodeBench)[reference:1]. It supports both thinking
# and non-thinking modes; we use non-thinking for development so the entire
# max_tokens window goes to visible output.
DEFAULT_DEVELOPMENT_MODEL = "qwen/qwen3-32b"

# Fallback model used when the primary development model hits a rate limit.
# Qwen3 32B has its own separate quota on Groq, so it makes a good fallback
# when the primary model is exhausted.
DEFAULT_FALLBACK_MODEL = "qwen/qwen3-32b"

_DEFAULT_MIN_REQUEST_INTERVAL = 1.0
_DEFAULT_MAX_RETRIES = 3

# Floors and caps used when adjusting max_tokens during retries.
_MIN_USEFUL_MAX_TOKENS = 4096
_MAX_ALLOWED_MAX_TOKENS = 32768


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


def get_development_model_name() -> str:
    """Non-reasoning model used for the code-generation phase."""
    try:
        if hasattr(st, "secrets") and "DEVELOPMENT_LLM_MODEL" in st.secrets:
            value = st.secrets["DEVELOPMENT_LLM_MODEL"]
            if value:
                return str(value).strip()
    except Exception:
        pass
    return os.getenv("DEVELOPMENT_LLM_MODEL", DEFAULT_DEVELOPMENT_MODEL).strip() or DEFAULT_DEVELOPMENT_MODEL


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


def _is_tool_use_failure(error: BaseException) -> bool:
    """Detect Groq rejecting GPT-OSS native tools in CrewAI's ReAct runtime."""
    text = str(error).lower()
    return "tool_use_failed" in text or "tool choice is none" in text


_REACT_TOOL_INSTRUCTION = {
    "role": "user",
    "content": (
        "Use CrewAI's text-based ReAct protocol for tools. Do not emit native "
        "JSON function calls, tool_calls, or a JSON object with name/arguments. "
        "If a tool is needed, output exactly:\n"
        "Thought: <brief reasoning>\n"
        "Action: <exact tool name>\n"
        "Action Input: <valid JSON object>\n"
        "Then wait for the tool result."
    ),
}

_DIRECT_OUTPUT_INSTRUCTION = {
    "role": "user",
    "content": (
        "Return the requested answer directly. Do not spend the entire "
        "response on hidden reasoning."
    ),
}


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

    def call(
        self,
        messages: list[dict[str, str]],
        callbacks: list[Any] | None = None,
    ) -> str:
        retries = _env_int("GROQ_MAX_RETRIES", _DEFAULT_MAX_RETRIES)
        request_messages = messages
        for attempt in range(retries + 1):
            _pace_requests()

            logger.warning(
                "Groq call attempt=%s model=%s max_tokens=%s reasoning_effort=%s",
                attempt,
                self.model,
                self.max_tokens,
                self.kwargs.get("reasoning_effort"),
            )

            try:
                response = super().call(request_messages, callbacks or [])

                logger.warning(
                    "Groq response type=%s len=%s repr=%r",
                    type(response).__name__,
                    len(str(response)) if response is not None else 0,
                    (str(response)[:120] if response is not None else None),
                )

                if response is not None and str(response).strip():
                    return response

                # -----------------------------------------------------------
                # EMPTY RESPONSE HANDLING
                # -----------------------------------------------------------
                # On reasoning models (gpt-oss-120b), an empty completion means
                # the hidden reasoning stream consumed the entire max_tokens
                # window. The reliable fix is to switch to a non-reasoning
                # model for the remainder of this call's retries. Qwen3 32B
                # supports a non-thinking mode that we can use.
                if attempt >= retries:
                    raise ValueError(
                        "Groq returned an empty response after "
                        f"{retries + 1} attempts."
                    )

                non_reasoning = _groq_model(get_development_model_name())
                if self.model != non_reasoning:
                    logger.warning(
                        "Empty response from %s; switching to non-reasoning model %s",
                        self.model,
                        non_reasoning,
                    )
                    self.model = non_reasoning
                    # Non-thinking mode for Qwen3: set reasoning_effort="none"
                    # or remove it entirely.
                    self.kwargs["reasoning_effort"] = "none"
                    self.max_tokens = max(
                        _MIN_USEFUL_MAX_TOKENS,
                        min(self.max_tokens or 16384, 16384),
                    )
                else:
                    # Already on a non-reasoning model; nudge the model out of
                    # whatever is causing the empty completion.
                    self.kwargs["reasoning_effort"] = "none"
                    current = self.max_tokens or 0
                    self.max_tokens = min(
                        _MAX_ALLOWED_MAX_TOKENS,
                        max(_MIN_USEFUL_MAX_TOKENS, current * 2),
                    )
                    if _DIRECT_OUTPUT_INSTRUCTION not in request_messages:
                        request_messages = [
                            *request_messages,
                            _DIRECT_OUTPUT_INSTRUCTION,
                        ]

                time.sleep(min(2.0, 0.5 * (attempt + 1)))
                continue

            except Exception as error:
                if _is_tool_use_failure(error) and attempt < retries:
                    if _REACT_TOOL_INSTRUCTION not in request_messages:
                        request_messages = [
                            *request_messages,
                            _REACT_TOOL_INSTRUCTION,
                        ]
                    time.sleep(0.25)
                    continue
                if not _is_retryable_rate_limit(error) or attempt >= retries:
                    raise
                if self.max_tokens:
                    self.max_tokens = max(
                        _MIN_USEFUL_MAX_TOKENS, self.max_tokens // 2
                    )

                fallback_model = _groq_model(get_fallback_model_name())
                switched_to_fallback = (
                    not getattr(self, "_fallback_used", False)
                    and self.model != fallback_model
                )
                if switched_to_fallback:
                    self.model = fallback_model
                    self._fallback_used = True
                    self.max_tokens = max(
                        _MIN_USEFUL_MAX_TOKENS,
                        min(self.max_tokens or _MIN_USEFUL_MAX_TOKENS, 1024),
                    )

                delay = _retry_delay(error, attempt)
                time.sleep(min(delay, 1.0) if switched_to_fallback else delay)

        raise RuntimeError("LLM call exhausted its retry budget")


# -------------------------------------------------------------------
# COMMON LLM FACTORY
# -------------------------------------------------------------------

def _create_llm(
    temperature: float = 0.3,
    max_tokens: int = 4096,
    model: str | None = None,
    *,
    reasoning_effort: str | None = "low",
) -> LLM:
    """Create a paced, retrying CrewAI LLM with a bounded output budget."""
    api_key = get_api_key("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = api_key

    kwargs: dict[str, Any] = {
        "model": _groq_model(model or get_model_name()),
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    return RateLimitAwareLLM(**kwargs)


# -------------------------------------------------------------------
# PUBLIC LLM FACTORIES
# -------------------------------------------------------------------

def get_llm(temperature: float = 0.3, max_tokens: int = 4096) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=max_tokens)


def get_planning_llm(temperature: float = 0.3) -> LLM:
    # Planning benefits from reasoning and is small enough to fit the budget.
    return _create_llm(temperature=temperature, max_tokens=4096)


def get_development_llm(temperature: float = 0.3) -> LLM:
    # Development writes many complete files in one response. Use Qwen3 32B
    # in non-thinking mode so the entire max_tokens window is available for
    # visible output instead of being consumed by hidden reasoning.
    # Qwen3 32B supports reasoning_effort="none" for non-thinking mode[reference:2].
    return _create_llm(
        temperature=temperature,
        max_tokens=16384,
        model=get_development_model_name(),
        reasoning_effort="none",
    )


def get_review_llm(temperature: float = 0.2) -> LLM:
    # Review/QA also produce long free-form reports; use the same non-reasoning
    # model so the full budget goes to visible output.
    return _create_llm(
        temperature=temperature,
        max_tokens=4096,
        model=get_development_model_name(),
        reasoning_effort="none",
    )


def get_debug_llm(temperature: float = 0.2) -> LLM:
    # Debug writes corrected files, so it needs the same treatment as dev.
    return _create_llm(
        temperature=temperature,
        max_tokens=8192,
        model=get_development_model_name(),
        reasoning_effort="none",
    )


def get_fast_llm(temperature: float = 0.3) -> LLM:
    return _create_llm(temperature=temperature, max_tokens=1024)


def get_powerful_llm(temperature: float = 0.3) -> LLM:
    # General-purpose "powerful" slot — reasoning is fine here.
    return _create_llm(temperature=temperature, max_tokens=8192)
