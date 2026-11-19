"""Voice input/output.

* Speech-to-text: Groq Whisper (``whisper-large-v3-turbo``) — Bangla + English.
* Voice commands: short spoken phrases ("clear chat", "উত্তর পড়ে শোনাও") are
  executed instead of being sent to the model as questions.
* Text-to-speech: the browser's built-in Web Speech API (free, no server cost).
"""

from __future__ import annotations

import enum
import json
import re
from typing import Any

from legalbot.llm import with_retry
from legalbot.text import detect_language

TRANSCRIPTION_PROMPT = (
    "Bangladesh law question. Penal Code, Code of Criminal Procedure, Constitution, "
    "Muslim Family Laws Ordinance, section, article, bail, ধারা, অনুচ্ছেদ, জামিন, দণ্ডবিধি।"
)


class VoiceError(ValueError):
    """Raised for unusable recordings (empty, too large, silent)."""


class VoiceCommand(enum.Enum):
    CLEAR_CHAT = "clear_chat"
    SHOW_SOURCES = "show_sources"
    HIDE_SOURCES = "hide_sources"
    READ_ANSWER = "read_answer"
    STOP_READING = "stop_reading"


_COMMAND_PHRASES: dict[VoiceCommand, tuple[str, ...]] = {
    VoiceCommand.CLEAR_CHAT: (
        "clear chat",
        "clear the chat",
        "clear conversation",
        "clear the conversation",
        "new chat",
        "reset chat",
        "start over",
        "delete chat",
        "চ্যাট মুছে ফেলো",
        "চ্যাট মুছে ফেলুন",
        "চ্যাট মুছুন",
        "নতুন চ্যাট",
        "কথোপকথন মুছে ফেলো",
        "কথোপকথন মুছে ফেলুন",
        "সব মুছে ফেলো",
    ),
    VoiceCommand.SHOW_SOURCES: (
        "show sources",
        "show the sources",
        "show source",
        "উৎস দেখাও",
        "উৎস দেখান",
        "সূত্র দেখাও",
    ),
    VoiceCommand.HIDE_SOURCES: (
        "hide sources",
        "hide the sources",
        "hide source",
        "উৎস লুকাও",
        "উৎস লুকান",
        "সূত্র লুকাও",
    ),
    VoiceCommand.READ_ANSWER: (
        "read the answer",
        "read answer",
        "read it",
        "read it aloud",
        "read aloud",
        "repeat the answer",
        "repeat answer",
        "উত্তর পড়ে শোনাও",
        "উত্তর পড়ে শোনান",
        "পড়ে শোনাও",
        "পড়ে শোনান",
        "উত্তরটা পড়ো",
    ),
    VoiceCommand.STOP_READING: (
        "stop",
        "stop reading",
        "stop speaking",
        "be quiet",
        "থামো",
        "থামুন",
        "পড়া বন্ধ করো",
    ),
}
_FILLERS = re.compile(
    r"^(?:please|okay|ok|hey|can you|could you|now|অনুগ্রহ করে|দয়া করে|প্লিজ)\s+|\s+(?:please|now|প্লিজ)$"
)
_PUNCT = re.compile(r"[.,!?;:।“”\"'’‘]+")


def _normalise(text: str) -> str:
    text = _PUNCT.sub(" ", text.lower())
    text = " ".join(text.split())
    for _ in range(3):  # strip stacked fillers: "okay please clear chat"
        stripped = _FILLERS.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    return text


_LOOKUP = {_normalise(p): cmd for cmd, phrases in _COMMAND_PHRASES.items() for p in phrases}


def parse_voice_command(transcript: str) -> VoiceCommand | None:
    """Return a command only when the WHOLE utterance is a command phrase."""
    return _LOOKUP.get(_normalise(transcript))


def resolve_input(typed: str, transcript: str = "") -> tuple[VoiceCommand | None, str]:
    """Decide what a chat submission means.

    Returns ``(command, "")`` when the submission is a voice/text command, or
    ``(None, question)`` otherwise. Typed text and a spoken transcript are
    combined into one question when both are present.
    """
    typed = (typed or "").strip()
    transcript = (transcript or "").strip()
    if typed and transcript:
        return None, f"{typed}\n{transcript}"
    single = typed or transcript
    if not single:
        return None, ""
    command = parse_voice_command(single)
    if command is not None:
        return command, ""
    return None, single


def transcribe(
    audio: bytes,
    *,
    api_key: str,
    model: str = "whisper-large-v3-turbo",
    filename: str = "audio.wav",
    language: str | None = None,
    max_mb: int = 20,
    client: Any = None,
) -> str:
    """Transcribe a recording with Groq Whisper. ``language``: "bn", "en" or None (auto)."""
    if not audio:
        raise VoiceError("The recording is empty. Please try again.")
    if len(audio) > max_mb * 1024 * 1024:
        raise VoiceError(f"The recording is larger than {max_mb} MB. Please keep it shorter.")
    if client is None:
        import groq

        client = groq.Groq(api_key=api_key, max_retries=0, timeout=60)

    params: dict[str, Any] = {
        "file": (filename or "audio.wav", audio),
        "model": model,
        "response_format": "json",
        "temperature": 0.0,
        "prompt": TRANSCRIPTION_PROMPT,
    }
    if language in {"bn", "en"}:
        params["language"] = language

    result = with_retry(lambda: client.audio.transcriptions.create(**params))
    text = getattr(result, "text", result if isinstance(result, str) else "")
    text = " ".join(str(text).split())
    if not text:
        raise VoiceError("No speech was detected. Please speak clearly and try again.")
    return text


def _plain_text_for_speech(markdown: str) -> str:
    text = re.sub(r"\[\d{1,2}\]", "", markdown)  # citation markers
    text = re.sub(r"[*_`#>|]+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links → label
    return " ".join(text.split())


def speech_html(text: str, language: str | None = None, *, stop_only: bool = False) -> str:
    """Self-contained HTML/JS snippet that speaks *text* via the Web Speech API.

    The text is embedded with ``json.dumps`` (and ``</`` escaped), so answer
    content can never break out of the script.
    """
    if stop_only:
        return "<script>try{window.parent.speechSynthesis.cancel()}catch(e){}</script>"
    spoken = _plain_text_for_speech(text)[:4000]
    lang = language or detect_language(spoken)
    bcp47 = "bn-BD" if lang == "bn" else "en-US"
    payload = json.dumps(spoken).replace("</", "<\\/")
    return f"""<script>
(function() {{
  try {{
    var synth = window.parent.speechSynthesis || window.speechSynthesis;
    var Utter = window.parent.SpeechSynthesisUtterance || window.SpeechSynthesisUtterance;
    if (!synth || !Utter) return;
    synth.cancel();
    var u = new Utter({payload});
    u.lang = "{bcp47}";
    var voices = synth.getVoices() || [];
    var prefix = "{bcp47[:2]}";
    for (var i = 0; i < voices.length; i++) {{
      if ((voices[i].lang || "").toLowerCase().indexOf(prefix) === 0) {{ u.voice = voices[i]; break; }}
    }}
    synth.speak(u);
  }} catch (e) {{}}
}})();
</script>"""
