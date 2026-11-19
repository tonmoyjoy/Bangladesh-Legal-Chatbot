"""Model failover: classification, ordering, streaming semantics, isolation, logging."""

from __future__ import annotations

import json
import logging

import groq
import httpx
import pytest
from langchain_core.documents import Document

from legalbot.config import ConfigError
from legalbot.llm import (
    AllModelsFailedError,
    FailureCategory,
    InvalidQuestionError,
    ModelFailover,
    PartialStreamError,
    build_chat_model,
    classify_error,
    friendly_error,
    model_order,
)
from legalbot.rag import RagPipeline, Turn
from tests.conftest import FailingChatModel, RecordingChatModel

SECRET_KEY = "gsk_SUPER_SECRET_KEY_123"
QUESTION = "What is the punishment for murder under section 302?"


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", "https://api.groq.com")
    )


def rate_limited() -> groq.RateLimitError:
    return groq.RateLimitError("rate limited", response=_resp(429), body=None)


def timed_out() -> groq.APITimeoutError:
    return groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


def disconnected() -> groq.APIConnectionError:
    return groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))


def bad_key() -> groq.AuthenticationError:
    return groq.AuthenticationError("invalid api key", response=_resp(401), body=None)


def not_found() -> groq.NotFoundError:
    return groq.NotFoundError("model not found", response=_resp(404), body=None)


# ── classification ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "exc, category",
    [
        (rate_limited(), FailureCategory.RATE_LIMIT),
        (timed_out(), FailureCategory.TIMEOUT),
        (disconnected(), FailureCategory.CONNECTION),
        (groq.InternalServerError("boom", response=_resp(503), body=None), FailureCategory.SERVER),
        (groq.APIStatusError("capacity", response=_resp(498), body=None), FailureCategory.SERVER),
        (groq.APIStatusError("too large", response=_resp(413), body=None), FailureCategory.RATE_LIMIT),
        (not_found(), FailureCategory.MODEL_UNAVAILABLE),
        (
            groq.PermissionDeniedError("blocked", response=_resp(403), body=None),
            FailureCategory.MODEL_UNAVAILABLE,
        ),
        (
            groq.BadRequestError(
                "x",
                response=_resp(400),
                body={
                    "error": {"code": "model_decommissioned", "message": "The model has been decommissioned"}
                },
            ),
            FailureCategory.MODEL_UNAVAILABLE,
        ),
        (
            groq.BadRequestError("x", response=_resp(400), body={"error": {"message": "messages: invalid"}}),
            FailureCategory.BAD_REQUEST,
        ),
        (bad_key(), FailureCategory.AUTH),
        (InvalidQuestionError("too long"), FailureCategory.INVALID_INPUT),
        (ConfigError("missing key"), FailureCategory.CONFIG),
        (ValueError("bug"), FailureCategory.UNKNOWN),
    ],
)
def test_classify_error(exc, category):
    assert classify_error(exc) is category


def test_retryable_vs_non_retryable_categories():
    next_model = {c for c in FailureCategory if c.try_next_model}
    assert next_model == {
        FailureCategory.RATE_LIMIT,
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION,
        FailureCategory.SERVER,
        FailureCategory.MODEL_UNAVAILABLE,
    }
    assert not FailureCategory.MODEL_UNAVAILABLE.transient  # no pointless same-model retries


# ── ordering ────────────────────────────────────────────────────────────────
def test_model_order_selected_first_deduplicated_and_retired_skipped():
    configured = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
    assert model_order("openai/gpt-oss-20b", configured) == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert model_order("llama-3.3-70b-versatile", configured) == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert model_order("a", ["a", "b"], unavailable={"a"}) == ["b"]
    assert model_order("only", ["only"]) == ["only"]  # one model = fallback disabled


# ── helpers ─────────────────────────────────────────────────────────────────
class Factory:
    """build(model) returning scripted fake chat models; records what was built."""

    def __init__(self, script):
        self.script = script
        self.built: list[str] = []

    def __call__(self, name):
        self.built.append(name)
        behaviour = self.script[name]
        if isinstance(behaviour, str):
            return RecordingChatModel(responses=[behaviour])
        if isinstance(behaviour, tuple):  # (tokens before failure, exception factory)
            tokens, exc = behaviour
            return FailingChatModel(after_tokens=list(tokens), failure=exc)
        return FailingChatModel(failure=behaviour)


def failover(script, models=("A", "B", "C"), **kwargs):
    factory = Factory(script)
    switches = []
    kwargs.setdefault("sleep", lambda s: None)
    runner = ModelFailover(
        models=list(models),
        build=factory,
        on_switch=lambda failed, cat, nxt: switches.append((failed, cat, nxt)),
        **kwargs,
    )
    return runner, factory, switches


def invoke_text(runner):
    return runner.invoke(lambda m: m.invoke("hi").content)


def stream_text(runner):
    return "".join(runner.stream(lambda m: (chunk.content for chunk in m.stream("hi"))))


# ── invoke ──────────────────────────────────────────────────────────────────
def test_fallback_from_first_to_second_model():
    runner, factory, switches = failover({"A": rate_limited, "B": "answer from B"})
    assert invoke_text(runner) == "answer from B"
    assert runner.used_model == "B" and runner.fallback_used
    assert factory.built == ["A", "B"]
    assert switches == [("A", FailureCategory.RATE_LIMIT, "B")]


@pytest.mark.parametrize(
    "exc",
    [bad_key, lambda: groq.BadRequestError("bad", response=_resp(400), body=None), lambda: ValueError("bug")],
)
def test_no_fallback_for_non_retryable_errors(exc):
    runner, factory, switches = failover({"A": exc, "B": "should not be used"})
    with pytest.raises(Exception) as caught:
        invoke_text(runner)
    assert not isinstance(caught.value, AllModelsFailedError)
    assert factory.built == ["A"] and switches == []


def test_all_models_failing_raises_with_attempts_and_backoff_on_last_model():
    sleeps = []
    runner, factory, _ = failover(
        {"A": rate_limited, "B": timed_out, "C": disconnected}, sleep=sleeps.append, last_model_attempts=3
    )
    with pytest.raises(AllModelsFailedError) as caught:
        invoke_text(runner)
    assert factory.built == ["A", "B", "C", "C", "C"]  # switch immediately; back off only on the last
    assert len(sleeps) == 2
    assert caught.value.categories == {
        FailureCategory.RATE_LIMIT,
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION,
    }
    assert "busy or unreachable" in friendly_error(caught.value)


def test_single_configured_model_disables_fallback_but_retries_transient_errors():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return rate_limited() if calls["n"] < 3 else None

    class Flaky(Factory):
        def __call__(self, name):
            self.built.append(name)
            err = flaky()
            return FailingChatModel(failure=err) if err else RecordingChatModel(responses=["ok"])

    factory = Flaky({})
    runner = ModelFailover(models=["only"], build=factory, sleep=lambda s: None)
    assert invoke_text(runner) == "ok"
    assert factory.built == ["only", "only", "only"]


# ── streaming ───────────────────────────────────────────────────────────────
def test_stream_falls_back_when_first_model_fails_before_any_token():
    runner, factory, switches = failover({"A": timed_out, "B": "streamed by B"})
    assert stream_text(runner) == "streamed by B"
    assert factory.built == ["A", "B"] and runner.used_model == "B"
    assert switches[0][1] is FailureCategory.TIMEOUT


def test_partial_stream_failure_is_not_concatenated_with_another_answer():
    runner, factory, _ = failover({"A": (["Murder is ", "punished "], disconnected), "B": "never"})
    emitted = []
    with pytest.raises(PartialStreamError) as caught:
        for token in runner.stream(lambda m: (c.content for c in m.stream("hi"))):
            emitted.append(token)
    assert "".join(emitted) == caught.value.partial_text == "Murder is punished "
    assert caught.value.model == "A" and caught.value.next_model == "B"
    assert caught.value.category is FailureCategory.CONNECTION
    assert factory.built == ["A"]  # B is offered for a retry, not silently appended


def test_later_calls_in_a_request_start_from_the_model_that_worked():
    runner, factory, _ = failover({"A": rate_limited, "B": "condensed"})
    invoke_text(runner)  # condense step: A fails, B works
    factory.script["B"] = "final answer"
    assert stream_text(runner) == "final answer"
    assert factory.built == ["A", "B", "B"]


def test_unavailable_models_are_skipped_for_the_session_only():
    session_a: set[str] = set()
    runner, _, _ = failover({"A": not_found, "B": "ok"}, unavailable=session_a)
    invoke_text(runner)
    assert session_a == {"A"}

    again, factory2, _ = failover({"A": "would work now", "B": "ok"}, unavailable=session_a)
    invoke_text(again)
    assert factory2.built == ["B"]  # same session: A not retried

    session_b: set[str] = set()
    other, factory3, _ = failover({"A": "fine", "B": "ok"}, unavailable=session_b)
    assert invoke_text(other) == "fine"
    assert factory3.built == ["A"]  # another user's session is unaffected


def test_no_candidates_left():
    runner, _, _ = failover({}, models=["A"], unavailable={"A"})
    with pytest.raises(AllModelsFailedError) as caught:
        invoke_text(runner)
    assert "No usable AI model" in friendly_error(caught.value)


def test_all_models_gone_message():
    runner, _, _ = failover({"A": not_found, "B": not_found}, models=["A", "B"])
    with pytest.raises(AllModelsFailedError) as caught:
        invoke_text(runner)
    assert "None of the configured AI models is available" in friendly_error(caught.value)


def test_logs_have_models_and_categories_but_no_secrets_or_questions(caplog):
    caplog.set_level(logging.INFO, logger="legalbot")
    runner, _, _ = failover({"A": rate_limited, "B": "ok"})
    runner.invoke(lambda m: m.invoke(f"{QUESTION} key={SECRET_KEY}").content)
    text = caplog.text
    assert "model=A" in text and "category=rate_limit" in text and "A -> B" in text
    assert "LLM success: model=B attempts=2 fallbacks=1" in text
    assert SECRET_KEY not in text and QUESTION not in text


# ── pipeline integration with mocked models ─────────────────────────────────
class StubRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, k):
        return self.docs[:k]


def murder_doc():
    return Document(
        page_content=(
            "Penal Code, 1860 — Section 302: Punishment for murder\nWhoever commits murder shall be punished."
        ),
        metadata={"act": "Penal Code, 1860", "section": "302", "chunk_id": "a"},
    )


def test_pipeline_fallback_keeps_history_and_validates_citations():
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    runner, factory, _ = failover(
        {"A": rate_limited, "B": "Standalone: punishment under section 302?"}, models=["A", "B"]
    )
    history = [Turn("user", "What is section 302?"), Turn("assistant", "It defines murder [1].")]
    prepared = pipeline.prepare("And the punishment?", history, runner, top_k=3)
    factory.script["B"] = "Death or life imprisonment [1][4]."
    text = "".join(pipeline.stream_answer(prepared, runner))
    result = pipeline.finalize(prepared, text, model=runner.used_model)
    assert result.text == "Death or life imprisonment [1]."
    assert result.invalid_citations == [4] and result.model == "B"
    assert result.sources == [murder_doc()]  # exactly what the model saw
    assert "What is section 302?" in prepared.history_text  # history intact after fallback


def test_partial_answer_is_marked_incomplete_and_still_validated():
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    runner, _, _ = failover({"A": (["Murder is punished [1][7]"], disconnected), "B": "x"}, models=["A", "B"])
    prepared = pipeline.prepare(QUESTION, [], runner, top_k=3)
    with pytest.raises(PartialStreamError) as caught:
        "".join(pipeline.stream_answer(prepared, runner))
    result = pipeline.finalize(prepared, caught.value.partial_text, incomplete=True, model="A")
    assert result.incomplete and result.needs_verification
    assert result.text == "Murder is punished [1]" and result.invalid_citations == [7]
    assert "interrupted" in result.warning and not result.refused


def test_condense_auth_error_is_not_swallowed():
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    runner, factory, _ = failover({"A": bad_key, "B": "x"}, models=["A", "B"])
    with pytest.raises(groq.AuthenticationError):
        pipeline.prepare("And then?", [Turn("user", "q"), Turn("assistant", "a")], runner, top_k=1)
    assert factory.built == ["A"]


# ── integration: real ChatGroq + real Groq SDK over a mocked HTTP transport ──
def _sse(model: str, text: str) -> bytes:
    events = []
    for i, token in enumerate(text.split(" ")):
        piece = token if i == 0 else " " + token
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}
            ],
        }
        events.append(f"data: {json.dumps(chunk)}\n\n")
    done = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    events.append(f"data: {json.dumps(done)}\n\n")
    events.append("data: [DONE]\n\n")
    return "".join(events).encode()


def test_real_groq_client_failover_over_mocked_http(caplog):
    """429 from the first model → second model streams; exercised through the real
    langchain-groq + groq SDK stack (only the network is mocked)."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        assert request.headers["authorization"] == f"Bearer {SECRET_KEY}"
        if body["model"] == "openai/gpt-oss-120b":
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Rate limit reached",
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers={"retry-after": "0"},
            )
        return httpx.Response(
            200,
            content=_sse(body["model"], "Murder is punishable by death [1][3]."),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ModelFailover(
        models=model_order("openai/gpt-oss-120b", ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]),
        build=lambda name: build_chat_model(SECRET_KEY, name, http_client=client),
        sleep=lambda s: None,
    )
    caplog.set_level(logging.DEBUG)
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    result = pipeline.answer(QUESTION, [], runner, top_k=3)

    assert [b["model"] for b in seen] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert seen[1]["stream"] is True and seen[1]["reasoning_effort"] == "low"
    assert result.text == "Murder is punishable by death [1]."
    assert result.model == "openai/gpt-oss-20b" and result.invalid_citations == [3]
    assert SECRET_KEY not in caplog.text
