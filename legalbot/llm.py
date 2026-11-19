"""Groq LLM construction, error classification, model failover and friendly errors.

Failover (``ModelFailover``) is the single place where "try another model"
logic lives; ``rag.py`` and ``app.py`` only call ``invoke`` / ``stream``.

* Order: the user's selected model first, then the other ``LLM_MODELS`` in
  order; retired models and models already found unavailable in this session
  are skipped.
* Retryable on the next model: rate limits, timeouts, connection errors,
  provider 5xx errors, model not found / disabled.
* Not retried: invalid API key, malformed request, empty/too-long input,
  configuration errors, unknown errors (bugs).
* Streaming: a failure before the first token moves to the next model; a
  failure after tokens were shown raises ``PartialStreamError`` (never
  concatenates a second answer onto a partial one).
* Logs carry model names, failure categories and counts only — never API
  keys, questions or answers.
"""

from __future__ import annotations

import enum
import logging
import random
import time
from collections.abc import Callable, Iterable, Iterator, MutableSet, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import groq
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from legalbot.config import RETIRED_LLM_MODELS, ConfigError
from legalbot.vectorstore import IncompatibleIndexError, IndexNotFoundError

logger = logging.getLogger(__name__)
T = TypeVar("T")

RETRYABLE: tuple[type[BaseException], ...] = (
    groq.RateLimitError,
    groq.APIConnectionError,
    groq.APITimeoutError,
    groq.InternalServerError,
)


class InvalidQuestionError(ValueError):
    """The question is empty or longer than the configured limit (not retried)."""


def build_chat_model(
    api_key: str,
    model: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    reasoning_effort: str = "low",
    http_client: Any = None,
) -> ChatGroq:
    """Create a per-request chat model (never shared between users)."""
    if not api_key:
        raise ConfigError("A Groq API key is required")
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": 0,  # retries/failover are handled in this module
        "timeout": 60,
    }
    # gpt-oss models are reasoning models; low effort keeps answers fast and
    # leaves the token budget for the actual answer.
    if model.startswith("openai/gpt-oss") and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if http_client is not None:
        kwargs["http_client"] = http_client
    return ChatGroq(**kwargs)


# ── error classification ────────────────────────────────────────────────────
class FailureCategory(enum.Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER = "server_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    AUTH = "invalid_api_key"
    BAD_REQUEST = "bad_request"
    INVALID_INPUT = "invalid_input"
    CONFIG = "configuration"
    UNKNOWN = "unknown"

    @property
    def try_next_model(self) -> bool:
        return self in _NEXT_MODEL

    @property
    def transient(self) -> bool:
        """Worth retrying the *same* model after a pause."""
        return self in _TRANSIENT


_TRANSIENT = frozenset(
    {FailureCategory.RATE_LIMIT, FailureCategory.TIMEOUT, FailureCategory.CONNECTION, FailureCategory.SERVER}
)
_NEXT_MODEL = _TRANSIENT | {FailureCategory.MODEL_UNAVAILABLE}
_MODEL_GONE_HINTS = (
    "decommissioned",
    "does not exist",
    "not found",
    "not supported",
    "not available",
    "blocked",
)


def _error_code_and_message(exc: BaseException) -> tuple[str, str]:
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or ""), str(error.get("message") or "")
    return "", str(getattr(exc, "message", "") or "")


def classify_error(exc: BaseException) -> FailureCategory:
    """Map an exception to a failure category (decides failover)."""
    if isinstance(exc, InvalidQuestionError):
        return FailureCategory.INVALID_INPUT
    if isinstance(exc, ConfigError):
        return FailureCategory.CONFIG
    if isinstance(exc, groq.APITimeoutError):  # subclass of APIConnectionError: check first
        return FailureCategory.TIMEOUT
    if isinstance(exc, groq.APIConnectionError):
        return FailureCategory.CONNECTION
    if isinstance(exc, groq.AuthenticationError):
        return FailureCategory.AUTH
    if isinstance(exc, groq.RateLimitError):
        return FailureCategory.RATE_LIMIT
    if isinstance(exc, groq.NotFoundError | groq.PermissionDeniedError):
        return FailureCategory.MODEL_UNAVAILABLE  # unknown model, or model blocked for this org
    if isinstance(exc, groq.InternalServerError):
        return FailureCategory.SERVER
    if isinstance(exc, groq.BadRequestError | groq.UnprocessableEntityError):
        code, message = _error_code_and_message(exc)
        lowered = f"{code} {message}".lower()
        if "model" in lowered and any(hint in lowered for hint in _MODEL_GONE_HINTS):
            return FailureCategory.MODEL_UNAVAILABLE  # e.g. code "model_decommissioned"
        return FailureCategory.BAD_REQUEST
    if isinstance(exc, groq.APIStatusError):
        if exc.status_code == 413:  # request larger than this model's per-minute token budget
            return FailureCategory.RATE_LIMIT
        if exc.status_code >= 498:  # 498 capacity exceeded, 5xx
            return FailureCategory.SERVER
        return FailureCategory.BAD_REQUEST
    return FailureCategory.UNKNOWN


# ── failover errors ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Attempt:
    model: str
    category: FailureCategory | None  # None = success


class AllModelsFailedError(RuntimeError):
    """Every candidate model failed with a retryable error."""

    def __init__(self, attempts: Sequence[Attempt]) -> None:
        self.attempts = list(attempts)
        categories = sorted({a.category.value for a in attempts if a.category is not None})
        super().__init__(f"All models failed ({', '.join(categories) or 'no models available'})")

    @property
    def categories(self) -> set[FailureCategory]:
        return {a.category for a in self.attempts if a.category is not None}


class PartialStreamError(RuntimeError):
    """A model failed after it had already streamed part of the answer."""

    def __init__(
        self, model: str, partial_text: str, category: FailureCategory, next_model: str | None
    ) -> None:
        super().__init__(f"Stream from {model} interrupted ({category.value})")
        self.model = model
        self.partial_text = partial_text
        self.category = category
        self.next_model = next_model


# ── model order ─────────────────────────────────────────────────────────────
def model_order(selected: str, configured: Iterable[str], unavailable: Iterable[str] = ()) -> list[str]:
    """Selected model first, then the others in configured order (deduplicated).

    Retired models and models found unavailable earlier in this session are skipped.
    """
    skip = set(unavailable) | set(RETIRED_LLM_MODELS)
    order: list[str] = []
    for name in (selected, *configured):
        name = name.strip()
        if name and name not in skip and name not in order:
            order.append(name)
    return order


# ── runners ─────────────────────────────────────────────────────────────────
class LLMRunner(Protocol):
    """What the RAG pipeline needs from "an LLM": one call or one stream."""

    @property
    def used_model(self) -> str: ...

    def invoke(self, call: Callable[[BaseChatModel], T]) -> T: ...

    def stream(self, call: Callable[[BaseChatModel], Iterator[str]]) -> Iterator[str]: ...


class SingleModelRunner:
    """One fixed model with same-model retry/backoff (behaviour of v2.0)."""

    def __init__(self, model: BaseChatModel, name: str = "") -> None:
        self.model = model
        self._name = name or str(getattr(model, "model_name", "") or type(model).__name__)

    @property
    def used_model(self) -> str:
        return self._name

    def invoke(self, call: Callable[[BaseChatModel], T]) -> T:
        return with_retry(lambda: call(self.model))

    def stream(self, call: Callable[[BaseChatModel], Iterator[str]]) -> Iterator[str]:
        return stream_with_retry(lambda: call(self.model))


def as_runner(llm: BaseChatModel | LLMRunner) -> LLMRunner:
    if isinstance(llm, BaseChatModel):
        return SingleModelRunner(llm)
    return llm


StatusCallback = Callable[[str, FailureCategory, str], None]


@dataclass
class ModelFailover:
    """Try models in order until one succeeds (see module docstring).

    ``build(model_name)`` returns a chat model. ``unavailable`` is a per-session
    set: models that are gone (not found / disabled) are added to it and skipped
    by later requests in the same session only.
    """

    models: Sequence[str]
    build: Callable[[str], BaseChatModel]
    unavailable: MutableSet[str] = field(default_factory=set)
    on_switch: StatusCallback | None = None
    last_model_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 20.0
    sleep: Callable[[float], None] = time.sleep
    attempts: list[Attempt] = field(default_factory=list)
    _start: int = 0
    _used: str = ""

    @property
    def used_model(self) -> str:
        return self._used

    @property
    def fallback_used(self) -> bool:
        return bool(self._used) and bool(self.models) and self._used != self.models[0]

    def _candidates(self) -> list[str]:
        return [m for m in self.models[self._start :] if m not in self.unavailable]

    def _next_after(self, model: str) -> str | None:
        """The next not-yet-unavailable model after *model* in the configured order."""
        names = list(self.models)
        start = names.index(model) + 1 if model in names else 0
        for name in names[start:]:
            if name not in self.unavailable:
                return name
        return None

    def _record_failure(self, model: str, exc: BaseException) -> FailureCategory:
        category = classify_error(exc)
        self.attempts.append(Attempt(model, category))
        logger.warning(
            "LLM attempt failed: model=%s category=%s error=%s attempt=%d",
            model, category.value, type(exc).__name__, len(self.attempts),
        )  # fmt: skip
        if category is FailureCategory.MODEL_UNAVAILABLE:
            self.unavailable.add(model)
        return category

    def _record_success(self, model: str) -> None:
        self.attempts.append(Attempt(model, None))
        self._used = model
        if model in self.models:
            self._start = list(self.models).index(model)  # later calls in this request start here
        fallbacks = len({a.model for a in self.attempts}) - 1
        logger.info(
            "LLM success: model=%s attempts=%d fallbacks=%d", model, len(self.attempts), max(fallbacks, 0)
        )

    def _switch(self, failed: str, category: FailureCategory) -> None:
        nxt = self._next_after(failed)
        if nxt is not None:
            logger.warning("LLM fallback: %s -> %s (reason=%s)", failed, nxt, category.value)
            if self.on_switch is not None:
                self.on_switch(failed, category, nxt)

    def _give_up(self) -> AllModelsFailedError:
        error = AllModelsFailedError(self.attempts)
        logger.error(
            "LLM failed: all models exhausted (attempts=%d, categories=%s)",
            len(self.attempts), ",".join(sorted(c.value for c in error.categories)) or "none",
        )  # fmt: skip
        return error

    def _plan(self) -> list[tuple[str, int]]:
        """(model, tries) pairs: one try each, extra backoff tries for the last model."""
        candidates = self._candidates()
        last = len(candidates) - 1
        return [(m, self.last_model_attempts if i == last else 1) for i, m in enumerate(candidates)]

    def invoke(self, call: Callable[[BaseChatModel], T]) -> T:
        for model, tries in self._plan():
            for attempt in range(tries):
                logger.info("LLM attempt: model=%s", model)
                try:
                    result = call(self.build(model))
                except Exception as exc:
                    category = self._record_failure(model, exc)
                    if not category.try_next_model:
                        raise
                    if category.transient and attempt + 1 < tries:
                        self.sleep(_delay(attempt, exc, self.base_delay, self.max_delay))
                        continue
                    self._switch(model, category)
                    break
                self._record_success(model)
                return result
        raise self._give_up()

    def stream(self, call: Callable[[BaseChatModel], Iterator[str]]) -> Iterator[str]:
        for model, tries in self._plan():
            for attempt in range(tries):
                logger.info("LLM attempt (stream): model=%s", model)
                emitted: list[str] = []
                try:
                    for token in call(self.build(model)):
                        emitted.append(token)
                        yield token
                except Exception as exc:
                    category = self._record_failure(model, exc)
                    if emitted:
                        raise PartialStreamError(
                            model, "".join(emitted), category, self._next_after(model)
                        ) from exc
                    if not category.try_next_model:
                        raise
                    if category.transient and attempt + 1 < tries:
                        self.sleep(_delay(attempt, exc, self.base_delay, self.max_delay))
                        continue
                    self._switch(model, category)
                    break
                self._record_success(model)
                return
        raise self._give_up()


# ── same-model retry helpers (single-model mode, voice transcription) ───────
def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _delay(attempt: int, exc: BaseException, base: float, cap: float) -> float:
    hinted = _retry_after(exc)
    if hinted is not None:
        return min(hinted, cap)
    backoff: float = min(cap, base * float(2**attempt))
    return backoff * (0.5 + random.random() / 2)


def with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call *fn*, retrying transient Groq errors with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except RETRYABLE as exc:
            if attempt == attempts - 1:
                raise
            wait = _delay(attempt, exc, base_delay, max_delay)
            logger.warning("Groq transient error (%s); retrying in %.1fs", type(exc).__name__, wait)
            sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def stream_with_retry(
    make_stream: Callable[[], Iterator[str]],
    *,
    attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Stream tokens; retry only if the failure happens before the first token."""
    for attempt in range(attempts):
        started = False
        try:
            for token in make_stream():
                started = True
                yield token
            return
        except RETRYABLE as exc:
            if started or attempt == attempts - 1:
                raise
            wait = _delay(attempt, exc, base_delay, max_delay)
            logger.warning("Groq transient error (%s); retrying in %.1fs", type(exc).__name__, wait)
            sleep(wait)


# ── user-facing messages ────────────────────────────────────────────────────
def friendly_error(exc: BaseException) -> str:
    """Message safe to show users (details go to the server log, never secrets)."""
    if isinstance(exc, AllModelsFailedError):
        if FailureCategory.MODEL_UNAVAILABLE in exc.categories and len(exc.categories) == 1:
            return (
                "None of the configured AI models is available. Check LLM_MODELS / GROQ_MODEL "
                "against the models listed at console.groq.com."
            )
        if not exc.attempts:
            return "No usable AI model is configured for this session. Reload the page or check LLM_MODELS."
        return "All configured AI models are busy or unreachable right now. Please try again in a minute."
    if isinstance(exc, InvalidQuestionError | ConfigError):
        return str(exc)
    if isinstance(exc, groq.AuthenticationError):
        return "The Groq API key was rejected. Check the key in the sidebar or server secrets."
    if isinstance(exc, groq.RateLimitError):
        return "The AI service is busy (rate limit reached). Please wait a minute and try again."
    if isinstance(exc, groq.APITimeoutError | groq.APIConnectionError):
        return "Could not reach the AI service. Check your internet connection and try again."
    if isinstance(exc, groq.NotFoundError):
        return "The selected model is not available on Groq any more. Pick another model in the sidebar."
    if isinstance(exc, groq.BadRequestError):
        return "The AI service rejected the request. Try a shorter question or another model."
    if isinstance(exc, IndexNotFoundError | IncompatibleIndexError):
        return str(exc)
    logger.error("Unexpected error while answering: %s", type(exc).__name__, exc_info=exc)
    return "Something went wrong while answering. Please try again."
