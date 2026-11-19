"""Shared test fixtures: offline fake embedder, sample statute pages, fixture PDF."""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legalbot.text import tokenize  # noqa: E402

DIM = 64


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    def __init__(self, name: str = "fake-embedder") -> None:
        self.name = name
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


PENAL_PAGES = [
    "THE PENAL CODE, 1860\n(ACT NO. XLV OF 1860)\n\nCHAPTER XVI\nOF OFFENCES AFFECTING THE HUMAN BODY\n"
    "Culpable homicide\n299. Whoever causes death by doing an act with the intention of causing death "
    "commits the offence of culpable homicide.\n"
    "300. Except in the cases hereinafter excepted, culpable homicide is murder, if the act by which "
    "the death is caused is done with the intention of causing death.",
    "Punishment for murder\n302. Whoever commits murder shall be punished with death, or imprisonment "
    "for life, and shall also be liable to fine.\n"
    "1. This line is a numbered list item inside a section and must not become a heading.\n"
    "CHAPTER XVII\nOF OFFENCES AGAINST PROPERTY\n"
    "378. Theft.—Whoever, intending to take dishonestly any movable property out of the possession "
    "of any person without that person's consent, moves that property, is said to commit theft.\n"
    "379. Punishment for theft.—Whoever commits theft shall be punished with imprisonment of either "
    "description for a term which may extend to three years, or with fine, or with both.",
]

CONSTITUTION_PAGES = [
    "PART III\nFUNDAMENTAL RIGHTS\n"
    "27. All citizens are equal before law and are entitled to equal protection of law.\n"
    "Article 33. Safeguards as to arrest and detention.—No person who is arrested shall be detained "
    "in custody without being informed of the grounds for such arrest, and every person arrested "
    "shall be produced before the nearest magistrate within twenty-four hours.",
]

BANGLA_PAGES = [
    "দণ্ডবিধি, ১৮৬০\nঅধ্যায় ১৬\n"
    "ধারা ৩০২। খুনের শাস্তি — যে ব্যক্তি খুন করিবে, সে মৃত্যুদণ্ডে বা যাবজ্জীবন কারাদণ্ডে দণ্ডিত হইবে "
    "এবং অর্থদণ্ডেও দণ্ডনীয় হইবে।\n"
    "ধারা ৩৭৯। চুরির শাস্তি — যে ব্যক্তি চুরি করিবে, সে তিন বৎসর পর্যন্ত কারাদণ্ডে বা অর্থদণ্ডে "
    "বা উভয় দণ্ডেই দণ্ডিত হইবে।",
]


def make_pages(texts: list[str], filename: str) -> list[Document]:
    return [
        Document(
            page_content=t,
            metadata={"source": filename, "filename": filename, "page": i, "total_pages": len(texts)},
        )
        for i, t in enumerate(texts, 1)
    ]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def penal_pages() -> list[Document]:
    return make_pages(PENAL_PAGES, "Penal code.pdf")


@pytest.fixture
def constitution_pages() -> list[Document]:
    return make_pages(CONSTITUTION_PAGES, "Constituion.pdf")


@pytest.fixture
def bangla_pages() -> list[Document]:
    return make_pages(BANGLA_PAGES, "penal_code_bangla.pdf")


@pytest.fixture
def all_chunks(penal_pages, constitution_pages, bangla_pages) -> list[Document]:
    from legalbot.chunker import chunk_documents

    return chunk_documents(penal_pages + constitution_pages + bangla_pages, chunk_size=600, chunk_overlap=60)


def write_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=9)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    write_pdf(data / "Penal code.pdf", PENAL_PAGES)
    write_pdf(data / "sub" / "Constituion.pdf", CONSTITUTION_PAGES)
    blank = pymupdf.open()
    blank.new_page()  # a page with no text → skipped
    blank.save(str(data / "blank.pdf"))
    blank.close()
    (data / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    return data


# ── fake chat model that records the prompts it receives ────────────────────
from typing import Any  # noqa: E402

from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402
from pydantic import Field  # noqa: E402


class RecordingChatModel(FakeListChatModel):
    seen: list[str] = Field(default_factory=list)

    def _record(self, messages: Any) -> None:
        self.seen.append("\n".join(str(m.content) for m in messages))

    def _call(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        self._record(messages)
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        self._record(messages)
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def make_llm():
    def factory(*responses: str) -> RecordingChatModel:
        return RecordingChatModel(responses=list(responses))

    return factory


class FailingChatModel(FakeListChatModel):
    """Raises ``failure`` (an exception, or a callable producing one) when called.

    With ``after_tokens`` set, streams those tokens first and then fails
    (simulates a provider dropping the connection mid-answer).
    """

    failure: Any = None
    after_tokens: list[str] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=lambda: ["unused"])

    def _exc(self) -> BaseException:
        return self.failure() if callable(self.failure) else self.failure

    def _call(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        raise self._exc()

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        for token in self.after_tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=token))
        raise self._exc()
