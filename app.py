"""Bangladesh Legal Aid Chatbot — Streamlit UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="বাংলাদেশ আইন সহায়তা | Bangladesh Legal Aid",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from legalbot import diagnostics, services, vectorstore, voice  # noqa: E402
from legalbot import llm as llm_mod  # noqa: E402
from legalbot.chunker import suspicious_reason  # noqa: E402
from legalbot.config import ConfigError, Settings, load_settings  # noqa: E402
from legalbot.rag import Turn, describe_source  # noqa: E402
from legalbot.ratelimit import SlidingWindowLimiter  # noqa: E402
from legalbot.text import detect_language, escape_markdown  # noqa: E402
from legalbot.voice import VoiceCommand, VoiceError  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

SAMPLE_QUESTIONS = (
    "What are the fundamental rights guaranteed under the Constitution of Bangladesh?",
    "What is the punishment for theft under the Penal Code?",
    "What safeguards does Article 33 of the Constitution give an arrested person?",
    "When can bail be granted for a non-bailable offence under the CrPC?",
    "Can a husband marry again without permission under the Muslim Family Laws Ordinance?",
    "দণ্ডবিধির ৩০২ ধারায় খুনের শাস্তি কী?",
)
VOICE_LANGUAGES = {"auto": "Auto-detect", "bn": "বাংলা (Bangla)", "en": "English"}
VOICE_HELP = (
    "**Voice commands** (say or type exactly):\n"
    "- *clear chat* · *চ্যাট মুছে ফেলো*\n"
    "- *show sources* / *hide sources* · *উৎস দেখাও* / *উৎস লুকাও*\n"
    "- *read the answer* · *উত্তর পড়ে শোনাও*\n"
    "- *stop* · *থামো*\n\n"
    "Anything else you say is asked as a question."
)


# ── cached, shareable resources (no per-user data inside) ─────────────────────
def _cache_key(settings: Settings) -> str:
    return repr(replace(settings, groq_api_key=""))


@st.cache_resource(show_spinner="Loading the legal index and language models…")
def get_backend(cache_key: str, _settings: Settings) -> services.Backend:
    return services.build_backend(_settings)


@st.cache_data(ttl=30, show_spinner=False)
def get_index_status(cache_key: str, _settings: Settings) -> vectorstore.IndexStatus:
    return vectorstore.index_status(_settings)


@st.cache_data(ttl=30, show_spinner=False)
def get_index_diagnostics(
    cache_key: str, status: vectorstore.IndexStatus, _settings: Settings
) -> list[diagnostics.Diagnostic]:
    return diagnostics.index_diagnostics(_settings, status)


# ── session state ─────────────────────────────────────────────────────────────
def init_state(settings: Settings) -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("user_api_key", "")
    ss.setdefault("model", settings.llm_model)
    if ss["model"] not in settings.llm_models:
        ss["model"] = settings.llm_model
    ss.setdefault("top_k", settings.top_k)
    ss.setdefault("show_sources", True)
    ss.setdefault("read_aloud", False)
    ss.setdefault("voice_language", "auto")
    ss.setdefault("pending_question", None)
    ss.setdefault("pending_retry", None)
    ss.setdefault("flash", [])
    # Models found gone (not found / disabled) — skipped for THIS session only.
    ss.setdefault("unavailable_models", set())
    if "limiter" not in ss:
        ss["limiter"] = SlidingWindowLimiter(settings.rate_limit_per_minute)
    if "voice_limiter" not in ss:
        ss["voice_limiter"] = SlidingWindowLimiter(settings.rate_limit_per_minute * 2)
    # Widget values changed by voice commands are applied here, before the
    # widgets are created (Streamlit forbids changing them afterwards).
    for key, value in ss.pop("pending_updates", {}).items():
        ss[key] = value


def effective_api_key(settings: Settings) -> str:
    return str(st.session_state.get("user_api_key") or settings.groq_api_key or "")


def clear_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["pending_question"] = None
    st.session_state["pending_retry"] = None


def queue_question(question: str) -> None:
    st.session_state["pending_question"] = question


def queue_retry(index: int) -> None:
    """Re-ask the question behind an interrupted answer, with the suggested model."""
    msg = st.session_state["messages"][index]
    msg["retried"] = True
    st.session_state["pending_retry"] = {
        "index": index,
        "question": msg["question"],
        "model": msg["retry_model"],
    }


def history_before(messages: list[dict[str, Any]], end: int) -> list[Turn]:
    """Conversation turns for the LLM: failed and interrupted answers are left out."""
    return [
        Turn(m["role"], m["content"])
        for m in messages[:end]
        if not m.get("error") and not m.get("incomplete")
    ]


def make_failover(settings: Settings, first_model: str) -> llm_mod.ModelFailover:
    """Per-request model failover; the only shared state is this session's own set."""
    api_key = effective_api_key(settings)

    def build(name: str) -> Any:
        return llm_mod.build_chat_model(
            api_key,
            name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            reasoning_effort=settings.llm_reasoning_effort,
        )

    def on_switch(failed: str, category: llm_mod.FailureCategory, nxt: str) -> None:
        st.toast(f"The selected model is unavailable. Trying the next configured model ({nxt})…")

    return llm_mod.ModelFailover(
        models=llm_mod.model_order(first_model, settings.llm_models),
        build=build,
        unavailable=st.session_state["unavailable_models"],
        on_switch=on_switch,
    )


def flash(message: str) -> None:
    st.session_state["flash"].append(message)


# ── rendering ─────────────────────────────────────────────────────────────────
def speak(text: str, language: str | None = None, *, stop: bool = False) -> None:
    components.html(voice.speech_html(text, language, stop_only=stop), height=0)


def render_sources(sources: list[dict[str, Any]], cited: list[int]) -> None:
    """Exactly the chunks that were given to the model, numbered as the model saw them."""
    if not sources:
        return
    with st.expander(f"📚 Sources given to the model ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            mark = " · 📌 cited in the answer" if i in cited else ""
            flag = " · ⚠️ suspicious document" if src.get("suspicious") else ""
            st.markdown(f"**[{i}]** {escape_markdown(src['label'])}{mark}{flag}")
            snippet = src["text"] if len(src["text"]) <= 700 else src["text"][:700] + "…"
            st.text(snippet)


def render_assistant_extras(msg: dict[str, Any], index: int, is_last: bool) -> None:
    if msg.get("model") and msg.get("model") != msg.get("selected_model"):
        st.caption(f"ℹ️ Answered using {msg['model']}.")
    if msg.get("warning"):
        if msg.get("needs_verification"):
            st.warning(f"⚠️ {msg['warning']}")
        else:
            st.caption(f"⚠️ {msg['warning']}")
    if st.session_state["show_sources"]:
        render_sources(msg.get("sources", []), msg.get("cited", []))
    if msg.get("incomplete") and not msg.get("retried") and msg.get("retry_model"):
        st.button(
            f"🔁 Retry with {msg['retry_model']}",
            key=f"retry_{index}",
            on_click=queue_retry,
            args=(index,),
        )
    if is_last and st.button("🔊 Read aloud", key=f"speak_{index}"):
        speak(msg["content"], msg.get("language"))


def render_message(msg: dict[str, Any], index: int, is_last: bool) -> None:
    avatar = "🧑" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("via_voice"):
            st.caption("🎙️ Voice input")
        if msg.get("retry_of") is not None:
            st.caption("🔁 Retry of the interrupted answer above")
        if msg.get("error"):
            st.error(msg["content"])
            return
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg, index, is_last)


def to_source_dicts(docs: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": describe_source(d),
            "text": d.page_content,
            "suspicious": bool(suspicious_reason(d.metadata)),
        }
        for d in docs
    ]


def render_diagnostic(diag: diagnostics.Diagnostic) -> None:
    show = st.error if diag.severity == diagnostics.ERROR else st.warning
    show(f"**{diag.message}**")
    st.caption("How to fix:")
    st.code(diag.fix, language="text")


# ── actions ───────────────────────────────────────────────────────────────────
def _question_index(messages: list[dict[str, Any]], answer_index: int) -> int:
    for i in range(answer_index - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return 0


def handle_question(
    question: str,
    settings: Settings,
    backend: services.Backend,
    *,
    via_voice: bool = False,
    retry_of: int | None = None,
    model: str | None = None,
) -> None:
    ss = st.session_state
    allowed, wait = ss["limiter"].try_acquire()
    if not allowed:
        st.warning(f"You're asking too quickly. Please wait {math.ceil(wait)} seconds.")
        if retry_of is not None:
            ss["messages"][retry_of]["retried"] = False  # keep the retry button
        return

    messages = ss["messages"]
    if retry_of is None:
        history = history_before(messages, len(messages))
        user_msg = {"role": "user", "content": question, "via_voice": via_voice}
        messages.append(user_msg)
        render_message(user_msg, len(messages) - 1, is_last=False)
    else:
        history = history_before(messages, _question_index(messages, retry_of))

    selected = model or ss["model"]
    failover = make_failover(settings, selected)
    language = detect_language(question)
    msg: dict[str, Any]
    with st.chat_message("assistant", avatar="⚖️"):
        if retry_of is not None:
            st.caption("🔁 Retry of the interrupted answer above")
        placeholder = st.empty()
        prepared = None
        try:
            with st.spinner("Searching the law…"):
                prepared = backend.pipeline.prepare(question, history, failover, ss["top_k"])
            with placeholder.container():
                streamed = st.write_stream(backend.pipeline.stream_answer(prepared, failover))
            raw = streamed if isinstance(streamed, str) else "".join(map(str, streamed))
            result = backend.pipeline.finalize(prepared, raw, model=failover.used_model)
            msg = answer_message(result, question, language, selected)
        except llm_mod.PartialStreamError as exc:
            if prepared is None:  # pragma: no cover - streaming only starts after prepare
                raise
            result = backend.pipeline.finalize(prepared, exc.partial_text, incomplete=True, model=exc.model)
            msg = answer_message(result, question, language, selected)
            msg["retry_model"] = exc.next_model or exc.model
            logger.warning("Answer interrupted mid-stream (category=%s)", exc.category.value)
        except Exception as exc:
            category = llm_mod.classify_error(exc)
            logger.warning("Answer failed (category=%s, error=%s)", category.value, type(exc).__name__)
            msg = {"role": "assistant", "content": llm_mod.friendly_error(exc), "error": True}
        if retry_of is not None:
            msg["retry_of"] = retry_of
        messages.append(msg)
        if msg.get("error"):
            placeholder.error(msg["content"])
        else:
            placeholder.markdown(msg["content"])
            render_assistant_extras(msg, len(messages) - 1, is_last=True)
    if ss["read_aloud"] and not msg.get("error") and not msg.get("incomplete"):
        speak(msg["content"], language)


def answer_message(result: Any, question: str, language: str, selected: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.text,
        "sources": to_source_dicts(result.sources),
        "cited": result.cited,
        "warning": result.warning,
        "needs_verification": result.needs_verification,
        "incomplete": result.incomplete,
        "language": language,
        "question": question,
        "model": result.model,
        "selected_model": selected,
    }


def run_command(command: VoiceCommand) -> None:
    ss = st.session_state
    if command is VoiceCommand.CLEAR_CHAT:
        clear_chat()
        flash("🧹 Conversation cleared.")
        st.rerun()
    elif command in (VoiceCommand.SHOW_SOURCES, VoiceCommand.HIDE_SOURCES):
        show = command is VoiceCommand.SHOW_SOURCES
        ss["pending_updates"] = {"show_sources": show}
        flash("📚 Sources shown." if show else "📚 Sources hidden.")
        st.rerun()
    elif command is VoiceCommand.READ_ANSWER:
        last = next(
            (m for m in reversed(ss["messages"]) if m["role"] == "assistant" and not m.get("error")),
            None,
        )
        if last is None:
            st.info("There is no answer to read yet.")
        else:
            speak(last["content"], last.get("language"))
            st.toast("🔊 Reading the last answer…")
    elif command is VoiceCommand.STOP_READING:
        speak("", stop=True)
        st.toast("🔇 Stopped.")


def transcribe_audio(audio: Any, settings: Settings) -> str | None:
    allowed, wait = st.session_state["voice_limiter"].try_acquire()
    if not allowed:
        st.warning(f"Too many recordings. Please wait {math.ceil(wait)} seconds.")
        return None
    language = st.session_state["voice_language"]
    try:
        with st.spinner("🎙️ Transcribing…"):
            transcript = voice.transcribe(
                audio.getvalue(),
                api_key=effective_api_key(settings),
                model=settings.transcription_model,
                filename=getattr(audio, "name", "") or "audio.wav",
                language=None if language == "auto" else language,
                max_mb=settings.max_audio_mb,
            )
    except VoiceError as exc:
        st.warning(str(exc))
        return None
    except Exception as exc:
        st.error(llm_mod.friendly_error(exc))
        return None
    st.toast(f"🎙️ Heard: {transcript}")
    return transcript


# ── page ──────────────────────────────────────────────────────────────────────
def render_sidebar(settings: Settings, status: vectorstore.IndexStatus) -> None:
    with st.sidebar:
        st.header("⚙️ Settings")
        if settings.groq_api_key:
            st.success("Server API key configured.")
            with st.expander("Use your own Groq key (this session only)"):
                st.text_input("Your Groq API key", type="password", key="user_api_key", placeholder="gsk_…")
        else:
            st.text_input(
                "Groq API key",
                type="password",
                key="user_api_key",
                placeholder="gsk_…",
                help="Kept only in your browser session. Get a free key at console.groq.com.",
            )
        st.selectbox("Model", settings.llm_models, key="model")
        st.slider("Sources per answer (Top-K)", min_value=1, max_value=15, key="top_k")
        st.toggle("Show sources", key="show_sources")

        st.subheader("🎙️ Voice")
        st.selectbox(
            "Speech language",
            list(VOICE_LANGUAGES),
            format_func=VOICE_LANGUAGES.__getitem__,
            key="voice_language",
        )
        st.toggle("Read answers aloud", key="read_aloud")
        with st.expander("Voice commands"):
            st.markdown(VOICE_HELP)

        st.button("🗑️ Clear conversation", on_click=clear_chat, width="stretch")

        st.subheader("📂 Legal index")
        if status.exists:
            st.success(f"{status.count:,} chunks indexed")
            if status.embedding_model:
                st.caption(f"Embeddings: `{status.embedding_model}`")
        elif status.error:
            st.error("Index unavailable")
        else:
            st.error("No index — run `python ingest.py`")
        st.caption(
            "**Stack:** PyMuPDF · FastEmbed (local multilingual embeddings) · ChromaDB · "
            "BM25 + reranker · Groq LLM & Whisper (cloud API) · Streamlit"
        )


def main() -> None:
    try:
        settings = load_settings(secrets=st.secrets)
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
        return

    init_state(settings)
    key = _cache_key(settings)
    status = get_index_status(key, settings)
    render_sidebar(settings, status)

    st.title("⚖️ বাংলাদেশ আইন সহায়তা")
    st.caption(
        "Bangladesh Legal Aid Chatbot — ask in Bangla or English, by text or voice. "
        "Answers are grounded in the loaded statutes and cite their sources."
    )
    for message in st.session_state["flash"]:
        st.toast(message)
    st.session_state["flash"] = []

    found = get_index_diagnostics(key, status, settings) + diagnostics.api_key_diagnostics(
        effective_api_key(settings)
    )
    for diag in found:
        render_diagnostic(diag)
    if any(d.severity == diagnostics.ERROR for d in found):
        st.stop()
        return

    try:
        backend = get_backend(key, settings)
    except Exception as exc:
        st.error(llm_mod.friendly_error(exc))
        st.stop()
        return

    if backend.suspicious_documents:
        st.warning(
            f"⚠️ The legal corpus contains {len(backend.suspicious_documents)} suspicious document(s) "
            "whose content does not match the file name. Answers citing them may be mislabelled."
        )
        with st.expander("Which documents?"):
            for name, reason in backend.suspicious_documents.items():
                st.markdown(f"- **{escape_markdown(name)}** — {escape_markdown(reason)}")

    messages = st.session_state["messages"]
    for i, msg in enumerate(messages):
        render_message(msg, i, is_last=i == len(messages) - 1)

    if not messages and not st.session_state["pending_question"]:
        st.markdown("#### 💡 Try a question")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            cols[i % 2].button(q, key=f"sample_{i}", on_click=queue_question, args=(q,), width="stretch")

    pending = st.session_state["pending_question"]
    if pending:
        st.session_state["pending_question"] = None
        handle_question(pending, settings, backend)

    retry = st.session_state["pending_retry"]
    if retry:
        st.session_state["pending_retry"] = None
        handle_question(retry["question"], settings, backend, retry_of=retry["index"], model=retry["model"])

    submission = st.chat_input(
        "Ask about Bangladesh law — type, or tap 🎙️ to speak…",
        accept_audio=True,
        audio_sample_rate=16000,
        max_chars=2000,
        key="chat_input",
    )
    if submission is not None:
        typed = submission if isinstance(submission, str) else str(getattr(submission, "text", "") or "")
        audio = None if isinstance(submission, str) else getattr(submission, "audio", None)
        transcript = ""
        if audio is not None:
            transcript = transcribe_audio(audio, settings) or ""
            if not transcript and not typed.strip():
                return
        command, question = voice.resolve_input(typed, transcript)
        if command is not None:
            run_command(command)
        elif question:
            handle_question(question, settings, backend, via_voice=audio is not None)

    st.caption(
        "⚖️ General legal information for education only — not a substitute for advice from a "
        "qualified advocate."
    )


main()
