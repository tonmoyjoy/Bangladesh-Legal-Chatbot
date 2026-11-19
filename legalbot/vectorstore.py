"""ChromaDB wrapper: local, self-hosted server, or Chroma Cloud.

* Chunk IDs are deterministic, so re-running ingest upserts instead of
  duplicating, and stale chunks of a re-ingested PDF are removed.
* The embedding model name is stored on the collection; loading the index
  with a different model fails loudly instead of returning garbage.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from legalbot.config import Settings
from legalbot.embeddings import Embedder

logger = logging.getLogger(__name__)

UPSERT_BATCH = 128
PAGE_SIZE = 1000


class IndexNotFoundError(RuntimeError):
    """The collection has not been built yet (run ingest)."""


class IncompatibleIndexError(RuntimeError):
    """The collection was built with a different embedding model."""


@dataclass(frozen=True)
class IndexStatus:
    exists: bool
    count: int = 0
    embedding_model: str = ""
    error: str = ""


def make_client(settings: Settings) -> Any:
    if settings.chroma_mode == "http":
        return chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
            headers={"x-chroma-token": settings.chroma_api_key} if settings.chroma_api_key else None,
        )
    if settings.chroma_mode == "cloud":
        return chromadb.CloudClient(
            tenant=settings.chroma_tenant or None,
            database=settings.chroma_database or None,
            api_key=settings.chroma_api_key,
        )
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_dir)


def _sanitize(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if isinstance(value, bool | int | float | str):
            clean[key] = value
        elif value is not None:
            clean[key] = str(value)
    return clean


def _check_compatible(collection: Any, model: str, name: str) -> None:
    built_with = str((collection.metadata or {}).get("embedding_model", ""))
    if built_with == model:
        return
    if not built_with and collection.count() == 0:
        return
    origin = f"'{built_with}'" if built_with else "an older version of this app"
    raise IncompatibleIndexError(
        f"The vector index '{name}' was built with {origin}, but EMBEDDING_MODEL is "
        f"'{model}'. Rebuild it with `python ingest.py --rebuild`."
    )


class VectorStore:
    def __init__(self, collection: Any, embedder: Embedder | None) -> None:
        self.collection = collection
        self.embedder = embedder

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def open(cls, client: Any, name: str, embedder: Embedder) -> VectorStore:
        """Open an existing collection (for querying)."""
        try:
            collection = client.get_collection(name=name, embedding_function=None)
        except Exception as exc:
            raise IndexNotFoundError(
                f"Vector index '{name}' not found. Run `python ingest.py` first."
            ) from exc
        _check_compatible(collection, embedder.name, name)
        if collection.count() == 0:
            raise IndexNotFoundError(f"Vector index '{name}' is empty. Run `python ingest.py`.")
        return cls(collection, embedder)

    @classmethod
    def create(cls, client: Any, name: str, embedder: Embedder, *, rebuild: bool = False) -> VectorStore:
        """Get-or-create a collection for ingestion (optionally wiping it first)."""
        if rebuild:
            try:
                client.delete_collection(name=name)
                logger.info("Deleted existing collection %s", name)
            except Exception as exc:  # collection did not exist yet
                logger.debug("Nothing to delete for %s: %s", name, exc)
        collection = client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.name},
        )
        _check_compatible(collection, embedder.name, name)
        return cls(collection, embedder)

    # ── writes ──────────────────────────────────────────────────────────
    def upsert(self, chunks: Sequence[Document]) -> int:
        if self.embedder is None:
            raise RuntimeError("An embedder is required to write to the index")
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            texts = [c.page_content for c in batch]
            self.collection.upsert(
                ids=[str(c.metadata["chunk_id"]) for c in batch],
                documents=texts,
                embeddings=self.embedder.embed_documents(texts),
                metadatas=[_sanitize(c.metadata) for c in batch],
            )
            logger.info("Upserted %d/%d chunks", min(i + UPSERT_BATCH, len(chunks)), len(chunks))
        return len(chunks)

    def remove_stale(self, source: str, keep_ids: Iterable[str]) -> int:
        """Delete chunks of *source* that are not in *keep_ids*."""
        keep = set(keep_ids)
        existing = self.collection.get(where={"source": source}, include=[])
        stale = [cid for cid in existing["ids"] if cid not in keep]
        if stale:
            self.collection.delete(ids=stale)
        return len(stale)

    # ── reads ───────────────────────────────────────────────────────────
    def count(self) -> int:
        return int(self.collection.count())

    def all_documents(self) -> list[Document]:
        """Every chunk (for the BM25 keyword index)."""
        docs: list[Document] = []
        total = self.count()
        for offset in range(0, total, PAGE_SIZE):
            page = self.collection.get(include=["documents", "metadatas"], limit=PAGE_SIZE, offset=offset)
            for cid, text, meta in zip(
                page["ids"], page["documents"] or [], page["metadatas"] or [], strict=False
            ):
                metadata = dict(meta or {})
                metadata["chunk_id"] = cid
                docs.append(Document(page_content=text or "", metadata=metadata))
        return docs

    def similarity_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.embedder is None:
            raise RuntimeError("An embedder is required to query the index")
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=max(1, min(k, self.count())),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[tuple[Document, float]] = []
        ids = result["ids"][0]
        texts = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for cid, text, meta, dist in zip(ids, texts, metas, dists, strict=False):
            metadata = dict(meta or {})
            metadata["chunk_id"] = cid
            hits.append((Document(page_content=text or "", metadata=metadata), float(dist)))
        return hits


def index_status(settings: Settings) -> IndexStatus:
    """Cheap status check that does not load the embedding model."""
    if settings.chroma_mode == "local" and not Path(settings.chroma_dir).exists():
        return IndexStatus(exists=False)
    try:
        client = make_client(settings)
        collection = client.get_collection(name=settings.collection_name, embedding_function=None)
        count = int(collection.count())
        model = str((collection.metadata or {}).get("embedding_model", ""))
        return IndexStatus(exists=count > 0, count=count, embedding_model=model)
    except Exception as exc:
        message = str(exc)
        if "does not exist" in message.lower() or "not found" in message.lower():
            return IndexStatus(exists=False)
        return IndexStatus(exists=False, error=message)
