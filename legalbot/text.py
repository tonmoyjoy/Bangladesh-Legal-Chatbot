"""Language helpers shared by chunking, retrieval and the UI."""

from __future__ import annotations

import re

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_BANGLA_CHAR = re.compile(r"[ঀ-৿]")
_LETTER = re.compile(r"[A-Za-zঀ-৿]")
# Bangla vowel signs are Unicode "marks", not letters, so \w alone would split
# words like "ধারা". Include the whole Bengali block explicitly.
_TOKEN = re.compile(r"[0-9a-zঀ-৿]+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "under",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def bn_to_ascii_digits(text: str) -> str:
    """Convert Bangla digits (০-৯) to ASCII digits."""
    return text.translate(BN_DIGITS)


def bangla_ratio(text: str) -> float:
    letters = _LETTER.findall(text)
    if not letters:
        return 0.0
    return len(_BANGLA_CHAR.findall(text)) / len(letters)


def detect_language(text: str) -> str:
    """Return ``"bn"`` for mostly-Bangla text, otherwise ``"en"``."""
    return "bn" if bangla_ratio(text) >= 0.3 else "en"


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def escape_markdown(text: str) -> str:
    """Escape Markdown/HTML-significant characters so text renders literally."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens for BM25 (Bangla-aware, digits normalised)."""
    normalised = bn_to_ascii_digits(text.lower())
    return [t for t in _TOKEN.findall(normalised) if t not in _STOPWORDS]
