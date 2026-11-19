"""RAG pipeline: condense follow-up → retrieve ONCE → answer (stream) → check citations.

The exact documents placed in the prompt are the ones returned to the UI, so
the sources panel always matches what the model saw.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from legalbot.citations import is_refusal, validate_citations
from legalbot.llm import (
    AllModelsFailedError,
    FailureCategory,
    InvalidQuestionError,
    LLMRunner,
    as_runner,
    classify_error,
)
from legalbot.prompts import (
    ANSWER_TEMPLATE,
    CONDENSE_SYSTEM,
    CONDENSE_TEMPLATE,
    REFUSAL_BN,
    REFUSAL_EN,
    SYSTEM_PROMPT,
)
from legalbot.retrieval import Retriever
from legalbot.text import detect_language

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class PreparedQuery:
    question: str
    standalone: str
    language: str
    docs: list[Document]
    context: str
    history_text: str


@dataclass
class AnswerResult:
    text: str
    sources: list[Document]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    refused: bool = False
    warning: str = ""
    standalone_question: str = ""
    incomplete: bool = False  # stream was interrupted; text is partial
    needs_verification: bool = False  # uncited or incomplete: user must verify
    model: str = ""  # model that produced the text (if known)


def describe_source(doc: Document) -> str:
    """'Penal Code, 1860 — Section 302 (Punishment for murder) — p. 120'."""
    meta = doc.metadata
    act = meta.get("act") or meta.get("filename") or "Unknown document"
    parts = [str(act)]
    if meta.get("section"):
        ref = f"{meta.get('unit', 'Section')} {meta['section']}"
        if meta.get("section_title"):
            ref += f" ({meta['section_title']})"
        parts.append(ref)
    page, page_end = meta.get("page"), meta.get("page_end")
    if page:
        parts.append(f"p. {page}" if not page_end or page_end == page else f"pp. {page}–{page_end}")
    return " — ".join(parts)


def format_docs(docs: Sequence[Document]) -> str:
    """Numbered context block: ``[n] <description>\\n<text>``."""
    return "\n\n".join(f"[{i}] {describe_source(doc)}\n{doc.page_content}" for i, doc in enumerate(docs, 1))


def format_history(history: Sequence[Turn], max_turns: int) -> str:
    if max_turns <= 0 or not history:
        return "(none)"
    recent = list(history)[-2 * max_turns :]
    lines = []
    for turn in recent:
        who = "User" if turn.role == "user" else "Assistant"
        content = turn.content if len(turn.content) <= 1200 else turn.content[:1200] + "…"
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def refusal_for(language: str) -> str:
    return REFUSAL_BN if language == "bn" else REFUSAL_EN


class RagPipeline:
    def __init__(self, retriever: Retriever, history_turns: int = 4) -> None:
        self.retriever = retriever
        self.history_turns = history_turns
        self._answer_prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", ANSWER_TEMPLATE)]
        )
        self._condense_prompt = ChatPromptTemplate.from_messages(
            [("system", CONDENSE_SYSTEM), ("human", CONDENSE_TEMPLATE)]
        )

    # ── steps ───────────────────────────────────────────────────────────
    def condense(self, question: str, history: Sequence[Turn], llm: BaseChatModel | LLMRunner) -> str:
        """Rewrite a follow-up into a standalone question (no-op without history)."""
        if not history or self.history_turns <= 0:
            return question
        payload = {"history": format_history(history, self.history_turns), "question": question}

        def call(model: BaseChatModel) -> str:
            chain: Any = self._condense_prompt | model | StrOutputParser()
            return str(chain.invoke(payload))

        try:
            rewritten = as_runner(llm).invoke(call).strip().strip('"')
        except AllModelsFailedError:
            logger.warning("Question condensing skipped: no model available; using the raw question")
            return question
        except Exception as exc:
            category = classify_error(exc)
            if category in (FailureCategory.AUTH, FailureCategory.CONFIG, FailureCategory.UNKNOWN):
                raise
            logger.warning("Question condensing failed (category=%s); using the raw question", category.value)
            return question
        if not rewritten or len(rewritten) > 3 * len(question) + 300:
            return question
        return rewritten

    def prepare(
        self, question: str, history: Sequence[Turn], llm: BaseChatModel | LLMRunner, top_k: int
    ) -> PreparedQuery:
        question = question.strip()
        if not question:
            raise InvalidQuestionError("Please type or say a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise InvalidQuestionError(
                f"The question is too long ({len(question):,} characters). "
                f"Please keep it under {MAX_QUESTION_CHARS:,}."
            )
        language = detect_language(question)
        standalone = self.condense(question, history, llm)
        docs = self.retriever.retrieve(standalone, top_k)
        return PreparedQuery(
            question=question,
            standalone=standalone,
            language=language,
            docs=docs,
            context=format_docs(docs),
            history_text=format_history(history, self.history_turns),
        )

    def stream_answer(self, prepared: PreparedQuery, llm: BaseChatModel | LLMRunner) -> Iterator[str]:
        """Stream the answer. With a ``ModelFailover`` runner, a failure before the
        first token moves to the next model; a failure mid-stream raises
        ``PartialStreamError`` (handled by the caller)."""
        if not prepared.docs:
            yield refusal_for(prepared.language)
            return
        payload = {
            "context": prepared.context,
            "history": prepared.history_text,
            "question": prepared.question,
            "language": "Bangla" if prepared.language == "bn" else "English",
        }

        def call(model: BaseChatModel) -> Iterator[str]:
            chain: Any = self._answer_prompt | model | StrOutputParser()
            return iter(chain.stream(payload))

        yield from as_runner(llm).stream(call)

    def finalize(
        self, prepared: PreparedQuery, raw_answer: str, *, incomplete: bool = False, model: str = ""
    ) -> AnswerResult:
        """Validate citations on the final (or partial) text.

        Citation numbers only ever refer to ``prepared.docs`` — the exact sources
        the model was given — so a citation never points at an unseen document.
        """
        raw_answer = raw_answer.strip()
        if not raw_answer and not incomplete:
            raw_answer = refusal_for(prepared.language)
        result = validate_citations(raw_answer, len(prepared.docs))
        refused = not incomplete and (is_refusal(result.text) or not prepared.docs)
        uncited = not refused and not result.cited
        notes = []
        if incomplete:
            notes.append("The answer was interrupted before it finished and may be incomplete.")
        if result.invalid:
            notes.append("Some citation numbers did not match a supplied source and were removed.")
        if uncited:
            notes.append("Needs verification: this answer cites none of the supplied sources.")
        return AnswerResult(
            text=result.text,
            sources=list(prepared.docs),
            cited=result.cited,
            invalid_citations=result.invalid,
            refused=refused,
            warning=" ".join(notes),
            standalone_question=prepared.standalone,
            incomplete=incomplete,
            needs_verification=incomplete or uncited,
            model=model,
        )

    # ── convenience (CLI / evals) ───────────────────────────────────────
    def answer(
        self, question: str, history: Sequence[Turn], llm: BaseChatModel | LLMRunner, top_k: int
    ) -> AnswerResult:
        runner = as_runner(llm)
        prepared = self.prepare(question, history, runner, top_k)
        text = "".join(self.stream_answer(prepared, runner))
        return self.finalize(prepared, text, model=runner.used_model)
