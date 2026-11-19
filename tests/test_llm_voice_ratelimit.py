from __future__ import annotations

import json
from types import SimpleNamespace

import groq
import httpx
import pytest

from legalbot.llm import build_chat_model, friendly_error, stream_with_retry, with_retry
from legalbot.ratelimit import SlidingWindowLimiter
from legalbot.vectorstore import IndexNotFoundError
from legalbot.voice import (
    VoiceCommand,
    VoiceError,
    parse_voice_command,
    resolve_input,
    speech_html,
    transcribe,
)


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status, headers=headers or {}, request=httpx.Request("POST", "https://api.groq.com")
    )


def rate_limit(headers=None) -> groq.RateLimitError:
    return groq.RateLimitError("slow down", response=_response(429, headers), body=None)


# ── LLM construction ────────────────────────────────────────────────────────
def test_build_chat_model_per_request():
    model = build_chat_model("gsk_a", "openai/gpt-oss-20b", temperature=0.2, max_tokens=512)
    assert model.model_name == "openai/gpt-oss-20b"
    assert model.groq_api_key.get_secret_value() == "gsk_a"
    assert model.max_retries == 0 and model.reasoning_effort == "low"
    other = build_chat_model("gsk_b", "some/other-model")
    assert other.groq_api_key.get_secret_value() == "gsk_b"
    assert other.reasoning_effort is None  # only gpt-oss gets reasoning_effort
    with pytest.raises(ValueError):
        build_chat_model("", "x")


# ── retry / backoff ─────────────────────────────────────────────────────────
def test_with_retry_recovers_and_honours_retry_after():
    sleeps, calls = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise rate_limit({"retry-after": "2"})
        return "ok"

    assert with_retry(flaky, sleep=sleeps.append) == "ok"
    assert sleeps == [2.0, 2.0]


def test_with_retry_gives_up_and_does_not_retry_auth_errors():
    sleeps = []
    with pytest.raises(groq.RateLimitError):
        with_retry(lambda: (_ for _ in ()).throw(rate_limit()), attempts=3, sleep=sleeps.append)
    assert len(sleeps) == 2

    auth = groq.AuthenticationError("bad key", response=_response(401), body=None)
    calls = []

    def bad():
        calls.append(1)
        raise auth

    with pytest.raises(groq.AuthenticationError):
        with_retry(bad, sleep=sleeps.append)
    assert len(calls) == 1


def test_stream_retries_only_before_first_token():
    attempts = []

    def make_stream():
        attempts.append(1)
        if len(attempts) == 1:
            raise rate_limit()
        yield "Hello "
        yield "world"

    assert "".join(stream_with_retry(make_stream, sleep=lambda s: None)) == "Hello world"
    assert len(attempts) == 2

    def breaks_midway():
        yield "partial"
        raise rate_limit()

    gen = stream_with_retry(breaks_midway, sleep=lambda s: None)
    assert next(gen) == "partial"
    with pytest.raises(groq.RateLimitError):
        next(gen)


def test_friendly_errors_hide_internals():
    assert "rate limit" in friendly_error(rate_limit())
    assert "API key" in friendly_error(groq.AuthenticationError("x", response=_response(401), body=None))
    assert "not available" in friendly_error(groq.NotFoundError("x", response=_response(404), body=None))
    assert "ingest" in friendly_error(IndexNotFoundError("Run `python ingest.py` first."))
    msg = friendly_error(RuntimeError("secret stack detail /home/user/.env"))
    assert "secret" not in msg and "went wrong" in msg


# ── rate limiter ────────────────────────────────────────────────────────────
def test_sliding_window_limiter():
    now = [0.0]
    limiter = SlidingWindowLimiter(2, 60, clock=lambda: now[0])
    assert limiter.try_acquire() == (True, 0.0)
    assert limiter.try_acquire()[0]
    allowed, wait = limiter.try_acquire()
    assert not allowed and wait == pytest.approx(60)
    now[0] = 61
    assert limiter.try_acquire()[0]


# ── voice commands ──────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("Clear chat.", VoiceCommand.CLEAR_CHAT),
        ("okay please clear the conversation", VoiceCommand.CLEAR_CHAT),
        ("চ্যাট মুছে ফেলো।", VoiceCommand.CLEAR_CHAT),
        ("Show sources", VoiceCommand.SHOW_SOURCES),
        ("উৎস লুকাও", VoiceCommand.HIDE_SOURCES),
        ("Read the answer please", VoiceCommand.READ_ANSWER),
        ("উত্তর পড়ে শোনাও", VoiceCommand.READ_ANSWER),
        ("Stop!", VoiceCommand.STOP_READING),
        ("থামো", VoiceCommand.STOP_READING),
        ("What is the punishment for theft?", None),
        ("How can I stop my husband from marrying again?", None),
        ("clear chat about bail law", None),
        ("", None),
    ],
)
def test_parse_voice_command(spoken, expected):
    assert parse_voice_command(spoken) is expected


def test_resolve_input():
    assert resolve_input("clear chat") == (VoiceCommand.CLEAR_CHAT, "")
    assert resolve_input("", "show sources") == (VoiceCommand.SHOW_SOURCES, "")
    assert resolve_input("", "জামিন কখন পাওয়া যায়?") == (None, "জামিন কখন পাওয়া যায়?")
    assert resolve_input("About bail:", "when is it granted") == (None, "About bail:\nwhen is it granted")
    assert resolve_input("  ", "") == (None, "")


# ── transcription (mocked Groq client) ──────────────────────────────────────
class FakeTranscriptions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def fake_client(*outcomes):
    tr = FakeTranscriptions(outcomes)
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=tr)), tr


def test_transcribe_sends_audio_and_language(monkeypatch):
    monkeypatch.setattr("legalbot.llm.time.sleep", lambda s: None)
    client, tr = fake_client(rate_limit(), "  ধারা ৩০২   কী? ")
    text = transcribe(b"RIFF....", api_key="k", language="bn", client=client, filename="rec.wav")
    assert text == "ধারা ৩০২ কী?"
    params = tr.calls[-1]
    assert params["file"] == ("rec.wav", b"RIFF....")
    assert params["model"] == "whisper-large-v3-turbo"
    assert params["language"] == "bn"
    assert len(tr.calls) == 2  # retried after 429

    client2, tr2 = fake_client("hello")
    transcribe(b"x", api_key="k", language="auto", client=client2)
    assert "language" not in tr2.calls[0]


def test_transcribe_rejects_bad_audio():
    with pytest.raises(VoiceError, match="empty"):
        transcribe(b"", api_key="k", client=object())
    with pytest.raises(VoiceError, match="larger"):
        transcribe(b"x" * (1024 * 1024 + 1), api_key="k", max_mb=1, client=object())
    client, _ = fake_client("   ")
    with pytest.raises(VoiceError, match="No speech"):
        transcribe(b"x", api_key="k", client=client)


# ── text-to-speech snippet ──────────────────────────────────────────────────
def test_speech_html_is_injection_safe_and_language_aware():
    nasty = 'Answer [1] **bold** </script><script>alert("x")</script>'
    html = speech_html(nasty, "en")
    assert html.count("</script>") == 1  # only our own closing tag
    assert "[1]" not in html and "**" not in html
    assert 'u.lang = "en-US"' in html
    bn = speech_html("খুনের শাস্তি মৃত্যুদণ্ড [1]।")
    assert 'u.lang = "bn-BD"' in bn
    payload = bn.split("new Utter(")[1].split(");")[0]
    assert json.loads(payload) == "খুনের শাস্তি মৃত্যুদণ্ড ।"
    assert "cancel()" in speech_html("", stop_only=True)
