"""Embedding back-ends.

Default: **FastEmbed** (ONNX Runtime, no PyTorch — small install, fast CPU
inference, deploys on Streamlit Cloud). Optional: **sentence-transformers**
(needs ``pip install -r requirements-torch.txt``).

The default model, ``intfloat/multilingual-e5-base``, understands Bangla and
English. E5 models need ``query:`` / ``passage:`` prefixes — handled here.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Models not in FastEmbed's built-in list, mapped to public ONNX exports.
# (repo, dim, onnx file)
CUSTOM_ONNX_MODELS: dict[str, tuple[str, int, str]] = {
    "intfloat/multilingual-e5-base": ("Xenova/multilingual-e5-base", 768, "onnx/model.onnx"),
    "intfloat/multilingual-e5-small": ("Xenova/multilingual-e5-small", 384, "onnx/model.onnx"),
}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def prefixes_for(model_name: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix) required by the model family."""
    lowered = model_name.lower()
    if "e5" in lowered and "mistral" not in lowered:
        return "query: ", "passage: "
    return "", ""


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding

        supported = {m["model"] for m in TextEmbedding.list_supported_models()}
        if model_name not in supported and model_name in CUSTOM_ONNX_MODELS:
            from fastembed.common.model_description import ModelSource, PoolingType

            repo, dim, model_file = CUSTOM_ONNX_MODELS[model_name]
            with contextlib.suppress(ValueError):  # already registered in this process
                TextEmbedding.add_custom_model(
                    model=model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=repo),
                    dim=dim,
                    model_file=model_file,
                )
        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)
        self._model: Any = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = [self.passage_prefix + t for t in texts]
        return [v.tolist() for v in self._model.embed(items, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        vec = next(iter(self._model.query_embed(self.query_prefix + text)))
        return [float(x) for x in vec]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "EMBEDDING_BACKEND=sentence-transformers needs: pip install -r requirements-torch.txt"
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        items = [self.passage_prefix + t for t in texts]
        vectors = self._model.encode(
            items, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([self.query_prefix + text], normalize_embeddings=True)[0]
        return list(map(float, vector))


def build_embedder(backend: str, model_name: str, cache_dir: str | None = None) -> Embedder:
    logger.info("Loading embedding model %s (%s)", model_name, backend)
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(model_name, cache_dir)
        except ImportError as exc:
            logger.warning(
                "sentence-transformers backend unavailable (%s); falling back to FastEmbed",
                exc,
            )
            return FastEmbedEmbedder(model_name, cache_dir)
    return FastEmbedEmbedder(model_name, cache_dir)
