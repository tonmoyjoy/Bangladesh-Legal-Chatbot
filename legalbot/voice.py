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
# Multi-word phrases may appear inside a short request ("can you read the answer
# for me"); single words such as "stop" must be the whole utterance.
_CONTAINED = sorted(
    ((p, cmd) for p, cmd in _LOOKUP.items() if len(p.split()) >= 2),
    key=lambda item: len(item[0]),
    reverse=True,
)
_MAX_COMMAND_WORDS = 10
# Words allowed around a command phrase ("can you read the answer for me now").
# Anything else ("clear chat about bail law") means it is not a command.
_POLITE_WORDS = frozenset(
    [
        "please",
        "can",
        "could",
        "would",
        "will",
        "you",
        "me",
        "for",
        "to",
        "now",
        "again",
        "just",
        "kindly",
        "okay",
        "ok",
        "hey",
        "i",
        "want",
        "like",
        "it",
        "out",
        "loud",
        "aloud",
        "the",
        "this",
        "that",
        "last",
        "latest",
        "আমাকে",
        "একটু",
        "প্লিজ",
        "দয়া",
        "অনুগ্রহ",
        "করে",
        "আবার",
        "এখন",
        "তো",
        "শেষ",
    ]
)

# Whisper's well-known outputs for silence / background noise.
_HALLUCINATIONS = frozenset(
    {
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thank you for watching",
        "you",
        "bye",
        "ধন্যবাদ",
        _normalise(TRANSCRIPTION_PROMPT),  # Whisper echoing its prompt
    }
)


def parse_voice_command(transcript: str) -> VoiceCommand | None:
    """Recognise a command phrase.

    Exact match first; otherwise a multi-word command phrase surrounded only by
    politeness words ("could you read the answer for me" → READ_ANSWER), so a
    real question that merely contains a phrase ("clear chat about bail law")
    is still asked as a question.
    """
    text = _normalise(transcript)
    exact = _LOOKUP.get(text)
    if exact is not None:
        return exact
    if not text or len(text.split()) > _MAX_COMMAND_WORDS:
        return None
    padded = f" {text} "
    for phrase, cmd in _CONTAINED:
        if f" {phrase} " in padded:
            leftover = padded.replace(f" {phrase} ", " ", 1).split()
            if all(word in _POLITE_WORDS for word in leftover):
                return cmd
    return None


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
    if not text or _normalise(text) in _HALLUCINATIONS:
        raise VoiceError("No speech was detected. Please speak clearly and try again.")
    return text


def _plain_text_for_speech(markdown: str) -> str:
    text = re.sub(r"\[\d{1,2}\]", "", markdown)  # citation markers
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links → label
    text = re.sub(r"[*_`#>|]+", " ", text)
    return " ".join(text.split())


def speech_chunks(text: str, limit: int = 180) -> list[str]:
    """Split text into sentence-sized pieces.

    Chrome silently stops a single utterance after ~15 seconds, so long answers
    are queued as several short utterances instead.
    """
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?।;:])\s+", text):
        sentence = sentence.strip()
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 3 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def speech_html(text: str, language: str | None = None, *, stop_only: bool = False, nonce: str = "") -> str:
    """HTML/JS snippet that speaks *text* with the browser's Web Speech API.

    Meant for ``st.html(..., unsafe_allow_javascript=True)`` (runs in the page
    itself, no iframe). ``nonce`` makes every call unique so Streamlit re-runs
    the script even when the same answer is read twice. The text is embedded
    with ``json.dumps`` (and ``</`` escaped), so answer content can never break
    out of the script.
    """
    tag = re.sub(r"[^A-Za-z0-9_-]", "", nonce)[:40]
    if stop_only:
        return f'<script data-nonce="{tag}">try{{window.speechSynthesis.cancel()}}catch(e){{}}</script>'
    spoken = _plain_text_for_speech(text)[:6000]
    lang = language or detect_language(spoken)
    bcp47 = "bn-BD" if lang == "bn" else "en-US"
    payload = json.dumps(speech_chunks(spoken), ensure_ascii=False).replace("</", "<\\/")
    note_id = f"tts-note-{tag or 'x'}"
    missing = (
        "আপনার ব্রাউজারে বাংলা ভয়েস নেই, তাই উত্তরটি পড়ে শোনানো যাচ্ছে না। "
        "Your browser has no Bangla voice. Try Chrome or Edge, or install a Bangla voice in your OS settings."
        if lang == "bn"
        else "Your browser does not support reading answers aloud."
    )
    return f"""<div id="{note_id}" style="display:none;font-size:0.85rem;opacity:0.8">🔇 {missing}</div>
<script data-nonce="{tag}">
(function() {{
  var note = document.getElementById("{note_id}");
  function fail() {{ if (note) note.style.display = "block"; }}
  try {{
    var synth = window.speechSynthesis;
    if (!synth || !window.SpeechSynthesisUtterance) {{ fail(); return; }}
    var parts = {payload};
    var started = false;
    function speak() {{
      if (started) return;
      started = true;
      var voices = synth.getVoices() || [];
      var prefix = "{bcp47[:2]}";
      var voice = null;
      for (var i = 0; i < voices.length; i++) {{
        var vlang = (voices[i].lang || "").toLowerCase().replace("_", "-");
        if (vlang.indexOf(prefix) === 0) {{ voice = voices[i]; break; }}
      }}
      if (!voice && prefix === "bn") {{ fail(); return; }}
      synth.cancel();
      for (var j = 0; j < parts.length; j++) {{
        var u = new SpeechSynthesisUtterance(parts[j]);
        u.lang = "{bcp47}";
        if (voice) u.voice = voice;
        synth.speak(u);
      }}
      if (synth.paused) synth.resume();
    }}
    if ((synth.getVoices() || []).length) {{ speak(); }}
    else {{
      synth.addEventListener("voiceschanged", speak, {{ once: true }});
      setTimeout(speak, 1500);  // some browsers never fire voiceschanged
    }}
  }} catch (e) {{ fail(); }}
}})();
</script>"""
