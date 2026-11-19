"""Pre-download the embedding and reranker models into MODEL_CACHE_DIR.

Run once after installing (or at Docker build time) so the first chat
request does not wait for a ~1.5 GB download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legalbot.config import load_settings
from legalbot.embeddings import build_embedder
from legalbot.retrieval import FastEmbedReranker


def main() -> int:
    settings = load_settings()
    print(f"Embedding model: {settings.embedding_model} → {settings.model_cache_dir}")
    embedder = build_embedder(settings.embedding_backend, settings.embedding_model, settings.model_cache_dir)
    dim = len(embedder.embed_query("ধারা ৩০২ — punishment for murder"))
    print(f"  ok (dimension {dim})")
    if settings.reranker_model:
        print(f"Reranker: {settings.reranker_model}")
        reranker = FastEmbedReranker(settings.reranker_model, settings.model_cache_dir)
        reranker.score("murder", ["Section 302 punishment for murder"])
        print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
