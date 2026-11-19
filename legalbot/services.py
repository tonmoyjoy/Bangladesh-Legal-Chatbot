"""Wire the heavy, shareable resources (models + index) into a pipeline.

Everything built here is safe to share between users: it holds no API keys
and no per-user settings. The UI caches it with ``st.cache_resource``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.documents import Document

from legalbot.chunker import suspicious_reason
from legalbot.config import Settings
from legalbot.embeddings import build_embedder
from legalbot.rag import RagPipeline
from legalbot.retrieval import FastEmbedReranker, HybridRetriever, Reranker
from legalbot.vectorstore import VectorStore, make_client

logger = logging.getLogger(__name__)


@dataclass
class Backend:
    pipeline: RagPipeline
    chunk_count: int
    embedding_model: str
    reranker_model: str
    # filename → reason, for documents whose content contradicts their file name
    suspicious_documents: dict[str, str] = field(default_factory=dict)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_model:
        return None
    try:
        return FastEmbedReranker(settings.reranker_model, settings.model_cache_dir)
    except Exception as exc:
        logger.warning("Reranker %s unavailable (%s); continuing without it", settings.reranker_model, exc)
        return None


def build_backend(settings: Settings) -> Backend:
    embedder = build_embedder(settings.embedding_backend, settings.embedding_model, settings.model_cache_dir)
    store = VectorStore.open(make_client(settings), settings.collection_name, embedder)
    corpus = store.all_documents()
    reranker = build_reranker(settings)
    retriever = HybridRetriever(store, corpus, reranker=reranker, candidate_k=settings.candidate_k)
    return Backend(
        pipeline=RagPipeline(retriever, history_turns=settings.history_turns, translate_queries=True),
        chunk_count=len(corpus),
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model if reranker is not None else "",
        suspicious_documents=find_suspicious(corpus),
    )


def find_suspicious(corpus: list[Document]) -> dict[str, str]:
    found: dict[str, str] = {}
    for doc in corpus:
        name = str(doc.metadata.get("filename") or doc.metadata.get("source") or "")
        if name in found:
            continue
        reason = suspicious_reason(doc.metadata)
        if reason:
            found[name] = reason
    if found:
        logger.warning("Corpus contains %d suspicious document(s): %s", len(found), ", ".join(found))
    return found
