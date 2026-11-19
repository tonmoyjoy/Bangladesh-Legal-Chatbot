from __future__ import annotations

import sys
import types

import pytest

from legalbot import embeddings, services
from legalbot.config import load_settings
from legalbot.embeddings import CUSTOM_ONNX_MODELS, FastEmbedEmbedder, build_embedder, prefixes_for
from legalbot.vectorstore import IndexNotFoundError, VectorStore, make_client
from tests.conftest import FakeEmbedder


def test_prefixes():
    assert prefixes_for("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert prefixes_for("BAAI/bge-m3") == ("", "")


def test_custom_onnx_models_register_with_real_fastembed(monkeypatch):
    from fastembed import TextEmbedding

    captured = {}

    class FakeModel:
        def embed(self, items, batch_size=32):
            captured["docs"] = list(items)
            return [types.SimpleNamespace(tolist=lambda: [0.1, 0.2]) for _ in captured["docs"]]

        def query_embed(self, text):
            captured["query"] = text
            return iter([[0.3, 0.4]])

    def fake_init(self, model_name, cache_dir=None, **kwargs):
        captured["model"] = model_name

    monkeypatch.setattr(TextEmbedding, "__init__", fake_init)
    emb = FastEmbedEmbedder("intfloat/multilingual-e5-base", cache_dir="/tmp/x")
    emb._model = FakeModel()
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    assert "intfloat/multilingual-e5-base" in supported  # registered via add_custom_model
    assert emb.embed_documents(["ধারা ৩০২"]) == [[0.1, 0.2]]
    assert captured["docs"] == ["passage: ধারা ৩০২"]
    assert emb.embed_query("murder") == [0.3, 0.4]
    assert captured["query"] == "query: murder"
    # registering twice in one process must not raise
    FastEmbedEmbedder("intfloat/multilingual-e5-base")
    assert set(CUSTOM_ONNX_MODELS) == {"intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small"}


def test_build_embedder_dispatch(monkeypatch):
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda name, cache: ("fast", name))
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", lambda name, cache: ("st", name))
    assert build_embedder("fastembed", "m") == ("fast", "m")
    assert build_embedder("sentence-transformers", "m") == ("st", "m")


def test_build_embedder_falls_back_from_missing_sentence_transformers(monkeypatch):
    def missing_backend(name, cache):
        raise ImportError("sentence-transformers is missing")

    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", missing_backend)
    monkeypatch.setattr(embeddings, "FastEmbedEmbedder", lambda name, cache: ("fast", name))
    assert build_embedder("sentence-transformers", "m") == ("fast", "m")


def test_build_backend_end_to_end(monkeypatch, tmp_path, all_chunks):
    settings = load_settings(
        env={"CHROMA_DIR": str(tmp_path / "db"), "EMBEDDING_MODEL": "fake-embedder", "RERANKER_MODEL": "r"}
    )
    embedder = FakeEmbedder()
    monkeypatch.setattr(services, "build_embedder", lambda *a: embedder)
    with pytest.raises(IndexNotFoundError):
        services.build_backend(settings)

    VectorStore.create(make_client(settings), settings.collection_name, embedder).upsert(all_chunks)

    class FakeReranker:
        def __init__(self, name, cache_dir):
            self.name = name

        def score(self, query, documents):
            return [float(len(d)) for d in documents]

    monkeypatch.setattr(services, "FastEmbedReranker", FakeReranker)
    backend = services.build_backend(settings)
    assert backend.chunk_count == len(all_chunks)
    assert backend.reranker_model == "r"
    docs = backend.pipeline.retriever.retrieve("punishment for theft", 3)
    assert len(docs) == 3


def test_reranker_failure_disables_it(monkeypatch):
    def boom(*a):
        raise OSError("download blocked")

    monkeypatch.setattr(services, "FastEmbedReranker", boom)
    assert services.build_reranker(load_settings(env={})) is None
    assert services.build_reranker(load_settings(env={"RERANKER_MODEL": ""})) is None


def test_sentence_transformers_backend_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="requirements-torch"):
        embeddings.SentenceTransformerEmbedder("x")
