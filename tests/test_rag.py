from __future__ import annotations

import pytest
from langchain_core.documents import Document

from legalbot.citations import is_refusal, validate_citations
from legalbot.prompts import REFUSAL_BN, REFUSAL_EN
from legalbot.rag import RagPipeline, Turn, describe_source, format_docs, format_history


class StubRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append((query, k))
        return self.docs[:k]


def murder_doc():
    return Document(
        page_content=(
            "Penal Code, 1860 — Section 302: Punishment for murder\n"
            "Whoever commits murder shall be punished with death."
        ),
        metadata={
            "act": "Penal Code, 1860",
            "unit": "Section",
            "section": "302",
            "section_title": "Punishment for murder",
            "page": 120,
            "page_end": 121,
            "chunk_id": "a",
        },
    )


# ── formatting ─────────────────────────────────────────────────────────────
def test_describe_source_and_format_docs():
    doc = murder_doc()
    assert describe_source(doc) == "Penal Code, 1860 — Section 302 (Punishment for murder) — pp. 120–121"
    plain = Document(page_content="x", metadata={"filename": "a.pdf", "page": 3})
    assert describe_source(plain) == "a.pdf — p. 3"
    block = format_docs([doc, plain])
    assert block.startswith("[1] Penal Code, 1860 — Section 302")
    assert "\n\n[2] a.pdf — p. 3\nx" in block


def test_format_history_limits_turns():
    history = [Turn("user", f"q{i}") if i % 2 == 0 else Turn("assistant", f"a{i}") for i in range(10)]
    text = format_history(history, max_turns=2)
    assert text.splitlines() == ["User: q6", "Assistant: a7", "User: q8", "Assistant: a9"]
    assert format_history([], 4) == "(none)"
    assert format_history(history, 0) == "(none)"


# ── citations ───────────────────────────────────────────────────────────────
def test_validate_citations_removes_invalid_numbers():
    result = validate_citations("Murder is punishable [1]. Also [7]. Mixed [1, 9][২].", n_sources=2)
    assert result.text == "Murder is punishable [1]. Also. Mixed [1][2]."
    assert result.cited == [1, 2]
    assert result.invalid == [7, 9]


def test_refusal_detection():
    assert is_refusal(REFUSAL_EN)
    assert is_refusal(REFUSAL_BN)
    assert not is_refusal("Section 302 says [1].")


# ── pipeline with a mocked LLM ──────────────────────────────────────────────
def test_single_retrieval_and_sources_match_prompt(make_llm):
    retriever = StubRetriever([murder_doc()])
    pipeline = RagPipeline(retriever)
    llm = make_llm("Murder is punished with death or life imprisonment [1].")
    prepared = pipeline.prepare("What is the punishment for murder?", [], llm, top_k=5)
    answer = "".join(pipeline.stream_answer(prepared, llm))
    result = pipeline.finalize(prepared, answer)

    assert len(retriever.queries) == 1  # retrieved exactly once
    assert result.sources == prepared.docs == [murder_doc()]
    assert result.cited == [1] and not result.warning and not result.refused
    prompt = llm.seen[0]
    assert "[1] Penal Code, 1860 — Section 302" in prompt
    assert "What is the punishment for murder?" in prompt
    assert "Answer ONLY from the numbered sources" in prompt


def test_follow_up_is_condensed_using_history(make_llm):
    retriever = StubRetriever([murder_doc()])
    pipeline = RagPipeline(retriever, history_turns=4)
    llm = make_llm("What is the punishment under Section 302 of the Penal Code?", "Death [1].")
    history = [Turn("user", "What does section 302 cover?"), Turn("assistant", "It covers murder [1].")]
    result = pipeline.answer("And the punishment?", history, llm, top_k=3)

    assert retriever.queries[0][0] == "What is the punishment under Section 302 of the Penal Code?"
    assert result.standalone_question.startswith("What is the punishment")
    assert "User: What does section 302 cover?" in llm.seen[0]  # condense prompt saw history
    assert "QUESTION (English):\nAnd the punishment?" in llm.seen[1]  # answer uses original question


def test_condense_failure_falls_back_to_original(make_llm):
    class Boom(StubRetriever):
        pass

    retriever = Boom([murder_doc()])
    pipeline = RagPipeline(retriever)
    llm = make_llm("")  # empty rewrite → keep original
    assert pipeline.condense("Original?", [Turn("user", "hi")], llm) == "Original?"


def test_no_documents_returns_refusal_without_calling_llm(make_llm):
    pipeline = RagPipeline(StubRetriever([]))
    llm = make_llm("should not be used")
    result = pipeline.answer("দণ্ডবিধিতে চুরির শাস্তি কী?", [], llm, top_k=5)
    assert result.text == REFUSAL_BN and result.refused
    assert llm.seen == []


def test_uncited_answer_gets_warning_and_bad_citation_removed(make_llm):
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    result = pipeline.answer("murder?", [], make_llm("Murder is serious."), top_k=5)
    assert "Needs verification" in result.warning and result.needs_verification
    result2 = pipeline.answer("murder?", [], make_llm("Death [1][4]."), top_k=5)
    assert result2.text == "Death [1]." and result2.invalid_citations == [4]
    assert "removed" in result2.warning


def test_empty_llm_output_becomes_refusal(make_llm):
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    result = pipeline.answer("murder?", [], make_llm(" "), top_k=5)
    assert result.refused and result.text == REFUSAL_EN


def test_bangla_question_flagged_in_prompt(make_llm):
    pipeline = RagPipeline(StubRetriever([murder_doc()]))
    llm = make_llm("মৃত্যুদণ্ড [1]।")
    result = pipeline.answer("খুনের শাস্তি কী?", [], llm, top_k=5)
    assert "QUESTION (Bangla)" in llm.seen[0]
    assert result.cited == [1]


def test_overlong_or_empty_question_is_rejected_not_truncated(make_llm):
    from legalbot.llm import InvalidQuestionError

    retriever = StubRetriever([murder_doc()])
    pipeline = RagPipeline(retriever)
    with pytest.raises(InvalidQuestionError, match="too long"):
        pipeline.prepare("x" * 5000, [], make_llm("ok"), top_k=1)
    with pytest.raises(InvalidQuestionError, match="question"):
        pipeline.prepare("   ", [], make_llm("ok"), top_k=1)
    assert retriever.queries == []  # nothing was retrieved or sent
