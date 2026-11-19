"""End-to-end UI tests with Streamlit's AppTest (no network, no models)."""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from legalbot import llm as llm_mod
from legalbot import services, vectorstore
from legalbot.rag import RagPipeline
from tests.conftest import FailingChatModel, RecordingChatModel

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class StubRetriever:
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query, k):
        return self.docs[:k]


@pytest.fixture
def ui(monkeypatch, all_chunks, tmp_path):
    """Patch the heavy backend + LLM; return a dict exposing what was used."""
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "LLM_MODELS", "RERANKER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "db"))
    monkeypatch.setenv("EMBEDDING_MODEL", "fake")
    monkeypatch.setattr("legalbot.config.load_dotenv", lambda *a, **k: False)
    st.cache_resource.clear()
    st.cache_data.clear()

    used = {"keys": [], "models": [], "answers": ["The punishment is death or life imprisonment [1]."]}
    theft = [c for c in all_chunks if c.metadata["section"] in {"302", "379"}]
    backend = services.Backend(
        pipeline=RagPipeline(StubRetriever(theft)),
        chunk_count=len(all_chunks),
        embedding_model="fake",
        reranker_model="",
    )
    monkeypatch.setattr(services, "build_backend", lambda settings: backend)
    monkeypatch.setattr(
        vectorstore,
        "index_status",
        lambda settings: vectorstore.IndexStatus(exists=True, count=len(all_chunks), embedding_model="fake"),
    )

    def fake_model(api_key, model, **kwargs):
        """Each built model answers with the next queued response (last one repeats)."""
        used["keys"].append(api_key)
        used["models"].append(model)
        failure = used.get("fail", {}).get(model)
        if isinstance(failure, dict):  # {"after_tokens": [...], "failure": exc_factory}
            return FailingChatModel(**failure)
        if failure is not None:
            return FailingChatModel(failure=failure)
        answer = used["answers"].pop(0) if len(used["answers"]) > 1 else used["answers"][0]
        return RecordingChatModel(responses=[answer])

    monkeypatch.setattr(llm_mod, "build_chat_model", fake_model)
    return used


def run_app(**session) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


def chat_texts(at: AppTest) -> list[str]:
    return [m.markdown[0].value for m in at.chat_message if m.markdown]


def code_blocks(at: AppTest) -> list[str]:
    return [c.value for c in at.code]


def test_requires_api_key(ui):
    at = run_app()
    assert not at.exception
    assert any("No Groq API key" in e.value for e in at.error)
    assert any("GROQ_API_KEY" in c for c in code_blocks(at))
    assert ui["keys"] == []


def test_missing_index_with_pdfs_shows_ingest_command(ui, monkeypatch, pdf_dir):
    monkeypatch.setenv("PDF_DIR", str(pdf_dir))
    monkeypatch.setattr(vectorstore, "index_status", lambda s: vectorstore.IndexStatus(exists=False))
    at = run_app(user_api_key="gsk_user")
    assert any("has not been built yet" in e.value for e in at.error)
    assert "python ingest.py" in code_blocks(at)
    assert not at.chat_input  # chat is blocked until the index exists


def test_missing_index_and_no_pdfs(ui, monkeypatch, tmp_path):
    monkeypatch.setenv("PDF_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(vectorstore, "index_status", lambda s: vectorstore.IndexStatus(exists=False))
    at = run_app(user_api_key="gsk_user")
    assert any("no PDFs" in e.value for e in at.error)
    assert any("data/README.txt" in c for c in code_blocks(at))


def test_incompatible_embedding_model_is_reported(ui, monkeypatch):
    monkeypatch.setattr(
        vectorstore,
        "index_status",
        lambda s: vectorstore.IndexStatus(exists=True, count=10, embedding_model="sentence-transformers/old"),
    )
    at = run_app(user_api_key="gsk_user")
    assert any("was built with 'sentence-transformers/old'" in e.value for e in at.error)
    assert any("--rebuild" in c for c in code_blocks(at))


def test_sample_question_gets_an_answer(ui):
    at = run_app(user_api_key="gsk_user")
    assert not at.exception
    sample = next(b for b in at.button if b.key == "sample_1")
    sample.click().run()
    assert not at.exception
    texts = chat_texts(at)
    assert texts[0] == "What is the punishment for theft under the Penal Code?"
    assert "death or life imprisonment [1]" in texts[1]
    assert ui["keys"] == ["gsk_user"]
    msgs = at.session_state["messages"]
    assert msgs[1]["cited"] == [1] and msgs[1]["sources"]


def test_chat_input_question_and_follow_up_history(ui):
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("What is the punishment for murder?").run()
    assert not at.exception
    ui["answers"] = ["Standalone question about murder", "It is death [1]."]
    at.chat_input(key="chat_input").set_value("And for attempt?").run()
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[-1]["content"] == "It is death [1]."


def test_per_session_settings_are_isolated(ui):
    a = run_app(user_api_key="gsk_alice", model="openai/gpt-oss-20b")
    a.chat_input(key="chat_input").set_value("theft?").run()
    b = run_app(user_api_key="gsk_bob")
    b.chat_input(key="chat_input").set_value("theft?").run()
    assert ui["keys"] == ["gsk_alice", "gsk_bob"]
    assert ui["models"] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def test_html_in_messages_is_not_rendered(ui):
    ui["answers"] = ["<img src=x onerror=alert(1)> safe [1]"]
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("<script>alert('x')</script>").run()
    assert not at.exception
    for el in at.markdown:
        assert getattr(el.proto, "allow_html", False) is False


def test_typed_voice_command_clears_chat(ui):
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("theft?").run()
    assert at.session_state["messages"]
    at.chat_input(key="chat_input").set_value("clear chat").run()
    assert not at.exception
    assert at.session_state["messages"] == []


def test_hide_sources_command_updates_toggle(ui):
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("hide sources").run()
    assert not at.exception
    assert at.session_state["show_sources"] is False
    assert at.toggle(key="show_sources").value is False


def test_rate_limit_blocks_spam(ui, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("q1").run()
    at.chat_input(key="chat_input").set_value("q2").run()
    assert any("too quickly" in w.value for w in at.warning)
    assert len(ui["keys"]) == 1


def test_llm_error_is_friendly(ui, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("internal traceback detail")

    monkeypatch.setattr(llm_mod, "build_chat_model", broken)
    at = run_app(user_api_key="gsk_user")
    at.chat_input(key="chat_input").set_value("theft?").run()
    assert not at.exception
    assert any("went wrong" in e.value for e in at.error)
    assert all("traceback" not in e.value for e in at.error)


# ── model failover in the UI ────────────────────────────────────────────────
import groq  # noqa: E402
import httpx  # noqa: E402

PRIMARY, SECONDARY = "openai/gpt-oss-120b", "openai/gpt-oss-20b"


def _resp(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))


def rate_limited():
    return groq.RateLimitError("rate limited", response=_resp(429), body=None)


def ask(at: AppTest, text: str) -> AppTest:
    return at.chat_input(key="chat_input").set_value(text).run()


def captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_fallback_answers_with_next_model_and_keeps_selection(ui):
    ui["fail"] = {PRIMARY: rate_limited}
    ui["answers"] = ["Death or life imprisonment [1][9]."]
    at = ask(run_app(user_api_key="gsk_user"), "Punishment for murder?")
    assert not at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "Death or life imprisonment [1]."  # citation [9] removed after fallback
    assert msgs[1]["model"] == SECONDARY
    assert any(f"Answered using {SECONDARY}" in c for c in captions(at))
    assert at.session_state["model"] == PRIMARY  # the user's selection is not changed
    assert ui["models"] == [PRIMARY, SECONDARY]


def test_invalid_api_key_is_not_retried_on_other_models(ui):
    ui["fail"] = {PRIMARY: lambda: groq.AuthenticationError("bad key", response=_resp(401), body=None)}
    at = ask(run_app(user_api_key="gsk_wrong"), "theft?")
    assert any("API key was rejected" in e.value for e in at.error)
    assert ui["models"] == [PRIMARY]
    assert all("gsk_wrong" not in e.value for e in at.error)


def test_all_models_failing_shows_friendly_error_and_keeps_history(ui):
    ui["fail"] = {PRIMARY: rate_limited, SECONDARY: rate_limited}
    at = ask(run_app(user_api_key="gsk_user"), "first question")
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["error"]
    assert any("All configured AI models are busy" in e.value for e in at.error)
    ui["fail"] = {}
    at = ask(at, "second question")
    msgs = at.session_state["messages"]
    assert [m["content"] for m in msgs if m["role"] == "user"] == ["first question", "second question"]
    assert not msgs[-1].get("error")


def test_partial_stream_offers_retry_with_next_model(ui):
    ui["fail"] = {PRIMARY: {"after_tokens": ["Murder is punished [1] and"], "failure": rate_limited}}
    ui["answers"] = ["The full answer: death or life imprisonment [1]."]
    at = ask(run_app(user_api_key="gsk_user"), "Punishment for murder?")
    assert not at.exception
    partial = at.session_state["messages"][1]
    assert partial["incomplete"] and partial["needs_verification"]
    assert partial["content"] == "Murder is punished [1] and"  # kept, never merged with another answer
    assert any("interrupted" in w.value for w in at.warning)
    assert ui["models"] == [PRIMARY]  # the second model is only used if the user asks

    retry = next(b for b in at.button if b.key == "retry_1")
    assert retry.label == f"🔁 Retry with {SECONDARY}"
    at = retry.click().run()
    assert not at.exception
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant"]  # question not re-entered
    assert msgs[1]["retried"] and msgs[2]["retry_of"] == 1
    assert msgs[2]["content"] == "The full answer: death or life imprisonment [1]."
    assert msgs[2]["model"] == SECONDARY and not msgs[2]["incomplete"]
    assert not any(b.key == "retry_1" for b in at.button)


def test_unavailable_model_is_isolated_per_session(ui):
    ui["fail"] = {PRIMARY: lambda: groq.NotFoundError("gone", response=_resp(404), body=None)}
    alice = ask(run_app(user_api_key="gsk_alice"), "q1")
    assert alice.session_state["unavailable_models"] == {PRIMARY}
    alice = ask(alice, "q2")  # with history: condense + answer, both on SECONDARY
    assert ui["models"][0] == PRIMARY and PRIMARY not in ui["models"][1:]  # Alice no longer tries it

    ui["fail"] = {}
    bob = ask(run_app(user_api_key="gsk_bob"), "q1")
    assert bob.session_state["unavailable_models"] == set()
    assert ui["models"][-1] == PRIMARY  # Bob's session still uses PRIMARY


def test_suspicious_corpus_warning(ui, monkeypatch):
    original = services.build_backend

    def with_suspicious(settings):
        backend = original(settings)
        backend.suspicious_documents = {"Muslim Family Law ordinance.pdf": "title mismatch"}
        return backend

    monkeypatch.setattr(services, "build_backend", with_suspicious)
    at = run_app(user_api_key="gsk_user")
    assert any("suspicious document" in w.value for w in at.warning)
