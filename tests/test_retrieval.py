from __future__ import annotations

import pytest

from legalbot.retrieval import HybridRetriever, explicit_references


class ListStore:
    """In-memory store returning dense hits in a fixed (deliberately poor) order."""

    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        return [(d, 0.5) for d in self.docs[:k]]


def sections(docs):
    return [d.metadata["section"] for d in docs]


def test_explicit_references():
    refs = explicit_references("What does Section 302 of the Penal Code say? Also art. 27")
    assert [(r.kind, r.num, r.act_hint) for r in refs] == [
        ("section", "302", "penal code"),
        ("article", "27", "penal code"),
    ]
    bn = explicit_references("দণ্ডবিধির ৩০২ ধারা নয়, ধারা ৩৭৯ কী?")
    assert [r.num for r in bn] == ["302", "379"]
    assert [r.num for r in explicit_references("৩৩ নং অনুচ্ছেদ")] == ["33"]
    assert explicit_references("অনুচ্ছেদ ৩৩")[0].kind == "article"
    assert explicit_references("no numbers here") == []


def test_bm25_finds_keyword_match_even_when_dense_is_poor(all_chunks):
    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, candidate_k=3)
    docs = retriever.retrieve("punishment for theft", k=3)
    assert "379" in sections(docs)


def test_exact_reference_is_pinned(all_chunks):
    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, candidate_k=2)
    docs = retriever.retrieve("Explain article 33 please", k=2)
    assert docs[0].metadata["section"] == "33"
    assert "Constitution" in docs[0].metadata["act"]


def test_bangla_reference_and_keywords(all_chunks):
    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, candidate_k=3)
    docs = retriever.retrieve("ধারা ৩০২ খুনের শাস্তি কী", k=3)
    assert any(d.metadata["section"] == "302" and d.metadata["unit"] == "ধারা" for d in docs)


def test_results_unique_and_limited(all_chunks):
    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, candidate_k=10)
    docs = retriever.retrieve("murder punishment death", k=4)
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == 4 == len(set(ids))


def test_reranker_reorders_but_keeps_pinned_first(all_chunks):
    class PreferTheft:
        def score(self, query, documents):
            return [10.0 if "theft" in d.lower() else 0.0 for d in documents]

    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, reranker=PreferTheft(), candidate_k=10)
    assert "theft" in retriever.retrieve("what is murder", k=1)[0].page_content.lower()
    pinned = retriever.retrieve("section 302 murder", k=2)
    assert pinned[0].metadata["section"] == "302"


def test_failing_reranker_is_not_fatal(all_chunks):
    class Broken:
        def score(self, query, documents):
            raise RuntimeError("model crashed")

    retriever = HybridRetriever(ListStore(all_chunks), all_chunks, reranker=Broken(), candidate_k=5)
    assert len(retriever.retrieve("theft", k=3)) == 3


def test_empty_query_and_corpus(all_chunks):
    retriever = HybridRetriever(ListStore(all_chunks), all_chunks)
    assert retriever.retrieve("   ", k=3) == []
    with pytest.raises(ValueError):
        HybridRetriever(ListStore([]), [])
