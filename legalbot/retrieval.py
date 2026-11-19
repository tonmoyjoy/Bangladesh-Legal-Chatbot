"""Hybrid retrieval: dense vectors + BM25 keywords → RRF fusion → reranker.

Also boosts exact references: a question mentioning "section 302",
"ধারা ৩০২" or "article 27" pulls those sections in directly.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from legalbot.text import bn_to_ascii_digits, tokenize

logger = logging.getLogger(__name__)

RRF_K = 60
_REF = re.compile(
    r"(?P<kw>section|sec\.?|s\.|article|art\.?|ধারা|অনুচ্ছেদ)\s*(?P<num>[0-9০-৯]{1,3}[a-zA-Z]?)\b",
    re.IGNORECASE,
)
# Bangla often puts the number first: "৩০২ ধারা", "৩৩ নং অনুচ্ছেদ".
_REF_BN_FIRST = re.compile(r"(?P<num>[0-9০-৯]{1,3})\s*(?:নং\s*)?(?P<kw>ধারা|অনুচ্ছেদ)")
_ACT_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"penal|দণ্ডবিধি|দন্ডবিধি", re.I), "penal code"),
    (re.compile(r"criminal procedure|crpc|cr\.p\.c|ফৌজদারি কার্যবিধি", re.I), "criminal procedure"),
    (re.compile(r"constitution|সংবিধান", re.I), "constitution"),
    (re.compile(r"muslim family|mflo|মুসলিম পারিবারিক", re.I), "muslim family"),
)


class Searchable(Protocol):
    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]: ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Document]: ...


class FastEmbedReranker:
    """Cross-encoder reranker via FastEmbed (ONNX, CPU)."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model_name
        self._model: Any = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(documents))]


@dataclass(frozen=True)
class Reference:
    kind: str  # "section" | "article"
    num: str
    act_hint: str


def explicit_references(query: str) -> list[Reference]:
    """Section/article numbers explicitly mentioned in the question."""
    act_hint = ""
    for pattern, hint in _ACT_HINTS:
        if pattern.search(query):
            act_hint = hint
            break
    found: list[tuple[int, Reference]] = []
    for pattern in (_REF, _REF_BN_FIRST):
        for m in pattern.finditer(query):
            kw = m.group("kw").lower()
            kind = "article" if kw.startswith("art") or kw == "অনুচ্ছেদ" else "section"
            ref = Reference(kind, bn_to_ascii_digits(m.group("num")).upper(), act_hint)
            if all(r != ref for _, r in found):
                found.append((m.start(), ref))
    return [ref for _, ref in sorted(found, key=lambda item: item[0])]


def _chunk_id(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id", id(doc)))


class HybridRetriever:
    def __init__(
        self,
        store: Searchable,
        corpus: Sequence[Document],
        reranker: Reranker | None = None,
        candidate_k: int = 20,
    ) -> None:
        if not corpus:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.store = store
        self.corpus = list(corpus)
        self.reranker = reranker
        self.candidate_k = candidate_k
        self._bm25 = BM25Okapi([tokenize(d.page_content) or ["_"] for d in self.corpus])

    def _keyword_hits(self, query: str, n: int) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        order = np.argsort(-scores)[:n]
        return [self.corpus[i] for i in order if scores[i] > 0]

    def _reference_hits(self, query: str) -> list[Document]:
        hits: list[Document] = []
        for ref in explicit_references(query):
            for doc in self.corpus:
                meta = doc.metadata
                if str(meta.get("section", "")).upper() != ref.num:
                    continue
                act = str(meta.get("act", "")).lower()
                is_constitution = "constitution" in act
                if ref.kind == "article" and not is_constitution:
                    continue
                if ref.kind == "section" and is_constitution:
                    continue
                if ref.act_hint and ref.act_hint not in act:
                    continue
                hits.append(doc)
        return hits

    def retrieve(self, query: str, k: int) -> list[Document]:
        query = query.strip()
        if not query:
            return []
        n = max(self.candidate_k, k)
        dense = [doc for doc, _ in self.store.similarity_search(query, n)]
        keyword = self._keyword_hits(query, n)
        pinned = self._reference_hits(query)

        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for ranked in (dense, keyword):
            for rank, doc in enumerate(ranked):
                cid = _chunk_id(doc)
                docs.setdefault(cid, doc)
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for doc in pinned:  # exact references always make the candidate list
            cid = _chunk_id(doc)
            docs.setdefault(cid, doc)
            scores[cid] = scores.get(cid, 0.0) + 1.0

        ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:n]
        candidates = [docs[cid] for cid in ordered]

        if self.reranker is not None and len(candidates) > 1:
            try:
                rerank = self.reranker.score(query, [d.page_content for d in candidates])
                pinned_ids = {_chunk_id(d) for d in pinned}
                candidates = [
                    doc
                    for _, doc in sorted(
                        zip(rerank, candidates, strict=True),
                        key=lambda pair: (_chunk_id(pair[1]) in pinned_ids, pair[0]),
                        reverse=True,
                    )
                ]
            except Exception as exc:  # reranking is an optimisation, never fatal
                logger.warning("Reranker failed, using fused order: %s", exc)
        return candidates[:k]
