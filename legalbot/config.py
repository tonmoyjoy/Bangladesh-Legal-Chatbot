"""Immutable application settings.

Settings are read once from (in priority order) environment variables, then
Streamlit secrets (if supplied by the caller), then defaults. The resulting
``Settings`` object is frozen: per-user choices (API key, model, Top-K) live in
the UI session, never in this shared object.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Groq production chat models (Nov 2026). llama-3.1-8b-instant and
# llama-3.3-70b-versatile were shut down for free/dev tiers on 2026-08-16.
DEFAULT_LLM_MODELS: tuple[str, ...] = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")

# Retired on Groq (free/dev tiers); old .env files may still name them.
RETIRED_LLM_MODELS = frozenset(
    {
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "qwen/qwen3-32b",
    }
)

VALID_CHROMA_MODES = ("local", "http", "cloud")
VALID_EMBEDDING_BACKENDS = ("fastembed", "sentence-transformers")


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclass(frozen=True)
class Settings:
    # LLM
    groq_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODELS[0]
    llm_models: tuple[str, ...] = DEFAULT_LLM_MODELS
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_reasoning_effort: str = "low"
    transcription_model: str = "whisper-large-v3-turbo"

    # Embeddings / reranking
    embedding_backend: str = "fastembed"
    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "BAAI/bge-reranker-v2-m3-int8"
    model_cache_dir: str = str(PROJECT_ROOT / ".cache" / "models")

    # Vector store
    chroma_mode: str = "local"
    chroma_dir: str = str(PROJECT_ROOT / "vectorstore" / "chroma_db")
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""
    collection_name: str = "bangladesh_law"

    # Ingestion
    pdf_dir: str = str(PROJECT_ROOT / "data")
    chunk_size: int = 1200
    chunk_overlap: int = 150
    ocr_lang: str = "eng+ben"
    ocr_cache_dir: str = str(PROJECT_ROOT / ".cache" / "ocr")
    ocr_workers: int = 0  # 0 = automatic
    allow_suspicious_docs: bool = False  # ingest PDFs whose title contradicts the file name

    # Retrieval / chat
    top_k: int = 5
    candidate_k: int = 20
    history_turns: int = 4

    # Abuse protection / voice
    rate_limit_per_minute: int = 8
    max_audio_mb: int = 20

    def validate(self) -> Settings:
        if self.chroma_mode not in VALID_CHROMA_MODES:
            raise ConfigError(f"CHROMA_MODE must be one of {VALID_CHROMA_MODES}, got {self.chroma_mode!r}")
        if self.embedding_backend not in VALID_EMBEDDING_BACKENDS:
            raise ConfigError(
                f"EMBEDDING_BACKEND must be one of {VALID_EMBEDDING_BACKENDS}, got {self.embedding_backend!r}"
            )
        if self.chroma_mode == "http" and not self.chroma_host:
            raise ConfigError("CHROMA_MODE=http requires CHROMA_HOST")
        if self.chroma_mode == "cloud" and not self.chroma_api_key:
            raise ConfigError("CHROMA_MODE=cloud requires CHROMA_API_KEY")
        if self.chunk_size < 200:
            raise ConfigError("CHUNK_SIZE must be >= 200")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
        if not self.llm_models:
            raise ConfigError("LLM_MODELS must list at least one model")

        top_k = min(max(self.top_k, 1), 15)
        candidate_k = max(self.candidate_k, top_k)
        llm_model = self.llm_model
        if llm_model in RETIRED_LLM_MODELS:
            logger.warning(
                "GROQ_MODEL=%s has been retired by Groq; using %s instead. Update your .env.",
                llm_model,
                DEFAULT_LLM_MODELS[0],
            )
            llm_model = DEFAULT_LLM_MODELS[0]
        llm_models = tuple(m for m in self.llm_models if m not in RETIRED_LLM_MODELS) or DEFAULT_LLM_MODELS
        if llm_model not in llm_models:
            llm_models = (llm_model, *llm_models)
        return replace(
            self,
            llm_model=llm_model,
            top_k=top_k,
            candidate_k=candidate_k,
            llm_models=llm_models,
            history_turns=max(self.history_turns, 0),
            rate_limit_per_minute=max(self.rate_limit_per_minute, 1),
            max_audio_mb=max(self.max_audio_mb, 1),
            ocr_workers=max(self.ocr_workers, 0),
        )


def _lookup(name: str, env: Mapping[str, str], secrets: Mapping[str, Any] | None) -> str | None:
    value = env.get(name)
    if value not in (None, ""):
        return str(value)
    if secrets is not None:
        try:
            secret = secrets.get(name)
        except Exception:  # Streamlit raises if no secrets file exists
            secret = None
        if secret not in (None, ""):
            return str(secret)
    return None


def _as_int(raw: str | None, default: int, name: str) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _as_float(raw: str | None, default: float, name: str) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(raw: str | None, default: str) -> str:
    if raw is None:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def load_settings(
    env: Mapping[str, str] | None = None,
    secrets: Mapping[str, Any] | None = None,
    *,
    load_env_file: bool = True,
) -> Settings:
    """Build validated settings from env vars / secrets / defaults."""
    if env is None:
        if load_env_file:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ
    d = Settings()

    def get(name: str) -> str | None:
        return _lookup(name, env, secrets)

    models_raw = get("LLM_MODELS")
    llm_models = tuple(m.strip() for m in models_raw.split(",") if m.strip()) if models_raw else d.llm_models
    reranker_raw = env.get("RERANKER_MODEL")
    if reranker_raw is None and secrets is not None:
        try:
            reranker_raw = secrets.get("RERANKER_MODEL")
        except Exception:
            reranker_raw = None
    # An explicitly empty RERANKER_MODEL disables reranking.
    reranker_model = d.reranker_model if reranker_raw is None else str(reranker_raw).strip()

    settings = Settings(
        groq_api_key=get("GROQ_API_KEY") or "",
        llm_model=get("GROQ_MODEL") or d.llm_model,
        llm_models=llm_models,
        llm_temperature=_as_float(get("LLM_TEMPERATURE"), d.llm_temperature, "LLM_TEMPERATURE"),
        llm_max_tokens=_as_int(get("LLM_MAX_TOKENS"), d.llm_max_tokens, "LLM_MAX_TOKENS"),
        llm_reasoning_effort=get("LLM_REASONING_EFFORT") or d.llm_reasoning_effort,
        transcription_model=get("TRANSCRIPTION_MODEL") or d.transcription_model,
        embedding_backend=(get("EMBEDDING_BACKEND") or d.embedding_backend).lower(),
        embedding_model=get("EMBEDDING_MODEL") or d.embedding_model,
        reranker_model=reranker_model,
        model_cache_dir=_resolve_path(get("MODEL_CACHE_DIR"), d.model_cache_dir),
        chroma_mode=(get("CHROMA_MODE") or d.chroma_mode).lower(),
        chroma_dir=_resolve_path(get("CHROMA_DIR"), d.chroma_dir),
        chroma_host=get("CHROMA_HOST") or "",
        chroma_port=_as_int(get("CHROMA_PORT"), d.chroma_port, "CHROMA_PORT"),
        chroma_ssl=_as_bool(get("CHROMA_SSL"), d.chroma_ssl),
        chroma_api_key=get("CHROMA_API_KEY") or "",
        chroma_tenant=get("CHROMA_TENANT") or "",
        chroma_database=get("CHROMA_DATABASE") or "",
        collection_name=get("COLLECTION_NAME") or d.collection_name,
        pdf_dir=_resolve_path(get("PDF_DIR"), d.pdf_dir),
        chunk_size=_as_int(get("CHUNK_SIZE"), d.chunk_size, "CHUNK_SIZE"),
        chunk_overlap=_as_int(get("CHUNK_OVERLAP"), d.chunk_overlap, "CHUNK_OVERLAP"),
        ocr_lang=get("OCR_LANG") or d.ocr_lang,
        ocr_cache_dir=_resolve_path(get("OCR_CACHE_DIR"), d.ocr_cache_dir),
        ocr_workers=_as_int(get("OCR_WORKERS"), d.ocr_workers, "OCR_WORKERS"),
        allow_suspicious_docs=_as_bool(get("ALLOW_SUSPICIOUS_DOCS"), d.allow_suspicious_docs),
        top_k=_as_int(get("TOP_K"), d.top_k, "TOP_K"),
        candidate_k=_as_int(get("CANDIDATE_K"), d.candidate_k, "CANDIDATE_K"),
        history_turns=_as_int(get("HISTORY_TURNS"), d.history_turns, "HISTORY_TURNS"),
        rate_limit_per_minute=_as_int(
            get("RATE_LIMIT_PER_MINUTE"), d.rate_limit_per_minute, "RATE_LIMIT_PER_MINUTE"
        ),
        max_audio_mb=_as_int(get("MAX_AUDIO_MB"), d.max_audio_mb, "MAX_AUDIO_MB"),
    )
    return settings.validate()
