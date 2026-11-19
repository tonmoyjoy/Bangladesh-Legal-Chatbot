from __future__ import annotations

from langchain_core.documents import Document

from evals.run_evals import (
    EvalItem,
    evaluate,
    first_relevant_rank,
    keyword_recall,
    load_dataset,
    parse_judge,
    to_markdown,
)
from legalbot.rag import AnswerResult


def doc(act, section):
    return Document(page_content=f"{act} {section}", metadata={"act": act, "section": section})


def test_dataset_is_well_formed():
    items = load_dataset()
    assert len(items) >= 25
    assert len({i.id for i in items}) == len(items)
    for item in items:
        if item.expect_refusal:
            assert not item.sections
        else:
            assert item.act and item.sections and item.keywords


def test_metric_helpers():
    item = EvalItem(id="x", question="q", act="penal", sections=["302"])
    docs = [doc("Constitution of Bangladesh", "302"), doc("Penal Code, 1860", "302")]
    assert first_relevant_rank(item, docs) == 2
    assert first_relevant_rank(item, docs[:1]) is None
    assert keyword_recall("Death or Imprisonment for life", ["death", "imprisonment for life"]) == 1.0
    assert keyword_recall("death", ["death", "fine"]) == 0.5
    assert parse_judge('Sure: {"score": 4, "reason": "ok"}') == 4.0
    assert parse_judge("no json") is None
    assert parse_judge('{"score": 9}') is None


def test_evaluate_end_to_end_with_stubs():
    items = [
        EvalItem(id="pc-302", question="murder?", act="penal", sections=["302"], keywords=["death"]),
        EvalItem(id="neg-x", question="football?", expect_refusal=True),
    ]

    def retrieve(question, k):
        return [doc("Penal Code, 1860", "302")] if "murder" in question else []

    def answer(question):
        if "murder" in question:
            return AnswerResult(text="Death [1].", sources=[doc("Penal Code, 1860", "302")], cited=[1])
        return AnswerResult(text="refused", sources=[], refused=True)

    _, summary = evaluate(items, retrieve, 5, answer, lambda ctx, ans: 5.0)
    assert summary["hit_at_k"] == 1.0 and summary["mrr"] == 1.0
    assert summary["keyword_recall"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["citation_validity"] == 1.0
    assert summary["faithfulness"] == 5.0
    table = to_markdown(summary)
    assert "| Retrieval hit@5 | 1.0 |" in table
