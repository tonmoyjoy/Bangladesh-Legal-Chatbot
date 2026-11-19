"""Offline evaluation of retrieval and answer quality.

Metrics
-------
* **hit@k** — the expected act + section is among the retrieved chunks.
* **MRR** — mean reciprocal rank of the first correct chunk.
* **keyword recall** — expected key phrases appear in the answer (``--answers``).
* **citation validity** — share of answers whose ``[n]`` markers all exist.
* **refusal accuracy** — out-of-scope questions are refused, in-scope are not.
* **faithfulness** — optional LLM-as-judge score 1-5 (``--judge``).

Usage:
    python -m evals.run_evals                 # retrieval metrics only (no API key needed)
    python -m evals.run_evals --answers       # + answer metrics (needs GROQ_API_KEY)
    python -m evals.run_evals --answers --judge
    python -m evals.run_evals --bm25-only     # keyword baseline straight from the PDFs
                                              # (no embedding model or index needed)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

DATASET = Path(__file__).with_name("dataset.jsonl")
RESULTS_MD = Path(__file__).with_name("results.md")
RESULTS_JSON = Path(__file__).with_name("results.json")

JUDGE_PROMPT = """You are grading a legal RAG answer for FAITHFULNESS.
Score 1-5: 5 = every claim is supported by the SOURCES; 1 = mostly unsupported.
Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}

SOURCES:
{context}

ANSWER:
{answer}"""


@dataclass
class EvalItem:
    id: str
    question: str
    act: str = ""
    sections: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    expect_refusal: bool = False


@dataclass
class ItemResult:
    id: str
    rank: int | None = None  # 1-based rank of first relevant chunk
    keyword_recall: float | None = None
    citations_valid: bool | None = None
    refused: bool | None = None
    refusal_correct: bool | None = None
    faithfulness: float | None = None


def load_dataset(path: Path = DATASET) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(EvalItem(**json.loads(line)))
    return items


def first_relevant_rank(item: EvalItem, docs: Sequence[Document]) -> int | None:
    for rank, doc in enumerate(docs, 1):
        act = str(doc.metadata.get("act", "")).lower()
        section = str(doc.metadata.get("section", "")).upper()
        if item.act in act and section in {s.upper() for s in item.sections}:
            return rank
    return None


def keyword_recall(answer: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for k in keywords if k.lower() in lowered) / len(keywords)


def parse_judge(raw: str) -> float | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        score = float(json.loads(match.group(0))["score"])
    except (ValueError, KeyError, TypeError):
        return None
    return score if 1 <= score <= 5 else None


def summarize(results: Sequence[ItemResult], k: int) -> dict[str, Any]:
    in_scope = [r for r in results if not r.id.startswith("neg-")]
    hits = [r for r in in_scope if r.rank is not None and r.rank <= k]
    mrr = sum(1.0 / r.rank for r in in_scope if r.rank) / len(in_scope) if in_scope else 0.0

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "k": k,
        "questions": len(results),
        "hit_at_k": round(len(hits) / len(in_scope), 3) if in_scope else None,
        "mrr": round(mrr, 3),
        "keyword_recall": mean([r.keyword_recall for r in in_scope if r.keyword_recall is not None]),
        "citation_validity": mean(
            [1.0 if r.citations_valid else 0.0 for r in results if r.citations_valid is not None]
        ),
        "refusal_accuracy": mean(
            [1.0 if r.refusal_correct else 0.0 for r in results if r.refusal_correct is not None]
        ),
        "faithfulness": mean([r.faithfulness for r in results if r.faithfulness is not None]),
    }


def evaluate(
    items: Sequence[EvalItem],
    retrieve: Callable[[str, int], list[Document]],
    k: int,
    answer_fn: Callable[[str], Any] | None = None,
    judge_fn: Callable[[str, str], float | None] | None = None,
) -> tuple[list[ItemResult], dict[str, Any]]:
    from legalbot.rag import format_docs

    results: list[ItemResult] = []
    for item in items:
        res = ItemResult(id=item.id)
        if not item.expect_refusal:
            res.rank = first_relevant_rank(item, retrieve(item.question, k))
        if answer_fn is not None:
            answer = answer_fn(item.question)
            res.refused = bool(answer.refused)
            res.refusal_correct = res.refused == item.expect_refusal
            res.citations_valid = not answer.invalid_citations
            if not item.expect_refusal:
                res.keyword_recall = keyword_recall(answer.text, item.keywords)
                if judge_fn is not None and not answer.refused:
                    res.faithfulness = judge_fn(format_docs(answer.sources), answer.text)
        results.append(res)
        print(f"  {item.id:14s} rank={res.rank}  refused={res.refused}", flush=True)
    return results, summarize(results, k)


def to_markdown(summary: dict[str, Any]) -> str:
    rows = [
        ("Questions", summary["questions"]),
        (f"Retrieval hit@{summary['k']}", summary["hit_at_k"]),
        ("Retrieval MRR", summary["mrr"]),
        ("Answer keyword recall", summary["keyword_recall"]),
        ("Citation validity", summary["citation_validity"]),
        ("Refusal accuracy", summary["refusal_accuracy"]),
        ("Faithfulness (LLM judge, 1-5)", summary["faithfulness"]),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {name} | {'—' if value is None else value} |" for name, value in rows]
    return "\n".join(lines) + "\n"


class _NoVectors:
    """Vector store stand-in for the keyword-only baseline."""

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return []


def bm25_retriever(candidate_k: int = 20) -> Any:  # pragma: no cover - needs PDFs
    from legalbot.chunker import chunk_documents
    from legalbot.config import load_settings
    from legalbot.pdf_loader import load_pdfs
    from legalbot.retrieval import HybridRetriever

    settings = load_settings()
    pages = load_pdfs(
        settings.pdf_dir,
        settings.ocr_lang,
        ocr_cache_dir=settings.ocr_cache_dir,
        ocr_workers=settings.ocr_workers or None,
    )
    chunks = chunk_documents(pages, settings.chunk_size, settings.chunk_overlap)
    return HybridRetriever(_NoVectors(), chunks, reranker=None, candidate_k=candidate_k)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs real models
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="also generate answers (needs GROQ_API_KEY)")
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness")
    parser.add_argument("--bm25-only", action="store_true", help="keyword baseline, no index needed")
    args = parser.parse_args(argv)

    if args.bm25_only:
        retriever = bm25_retriever()
        results, summary = evaluate(load_dataset(), retriever.retrieve, args.k)
        print(to_markdown(summary))
        return 0

    from legalbot.config import load_settings
    from legalbot.llm import build_chat_model, with_retry
    from legalbot.services import build_backend

    settings = load_settings()
    backend = build_backend(settings)
    pipeline = backend.pipeline
    answer_fn = judge_fn = None
    if args.answers or args.judge:
        if not settings.groq_api_key:
            print("GROQ_API_KEY is required for --answers/--judge", file=sys.stderr)
            return 2
        llm = build_chat_model(settings.groq_api_key, settings.llm_model, max_tokens=settings.llm_max_tokens)

        def answer_fn(question: str) -> Any:
            return pipeline.answer(question, [], llm, args.k)

        if args.judge:

            def judge_fn(context: str, answer: str) -> float | None:
                prompt = JUDGE_PROMPT.format(context=context, answer=answer)
                raw = with_retry(lambda: llm.invoke(prompt).content)
                return parse_judge(str(raw))

    results, summary = evaluate(load_dataset(), pipeline.retriever.retrieve, args.k, answer_fn, judge_fn)
    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "items": [asdict(r) for r in results]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    RESULTS_MD.write_text(to_markdown(summary), encoding="utf-8")
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
