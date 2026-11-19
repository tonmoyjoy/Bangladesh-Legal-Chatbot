from __future__ import annotations

import pytest

from legalbot.config import DEFAULT_LLM_MODELS, PROJECT_ROOT, ConfigError, load_settings
from legalbot.text import bn_to_ascii_digits, detect_language, escape_markdown, tokenize


# ── text helpers ─────────────────────────────────────────────────────────────
def test_detect_language():
    assert detect_language("দণ্ডবিধির ৩০২ ধারায় খুনের শাস্তি কী?") == "bn"
    assert detect_language("What is section 302?") == "en"
    assert detect_language("") == "en"
    assert detect_language("12345") == "en"


def test_tokenize_keeps_bangla_words_whole_and_normalises_digits():
    tokens = tokenize("ধারা ৩০২ খুনের শাস্তি")
    assert tokens == ["ধারা", "302", "খুনের", "শাস্তি"]
    assert tokenize("What is the punishment for Theft?") == ["punishment", "theft"]


def test_bn_digits():
    assert bn_to_ascii_digits("৩০২ক") == "302ক"


def test_escape_markdown_neutralises_html_and_markup():
    escaped = escape_markdown("<script>alert(1)</script> **bold** [x](y)")
    assert "<" not in escaped.replace("\\<", "")
    assert "\\*\\*bold\\*\\*" in escaped
    assert "\\[x\\]\\(y\\)" in escaped


# ── config ──────────────────────────────────────────────────────────────────
def test_defaults_without_env():
    s = load_settings(env={})
    assert s.groq_api_key == ""
    assert s.llm_model == DEFAULT_LLM_MODELS[0]
    assert s.embedding_backend == "fastembed"
    assert s.top_k == 5 and s.candidate_k >= s.top_k
    assert s.chroma_dir.startswith(str(PROJECT_ROOT))


def test_env_overrides_and_relative_paths():
    s = load_settings(
        env={
            "GROQ_API_KEY": "gsk_test",
            "GROQ_MODEL": "custom/model",
            "TOP_K": "99",
            "CANDIDATE_K": "3",
            "CHROMA_DIR": "./x/db",
            "LLM_MODELS": "a, b",
            "RERANKER_MODEL": "",
        }
    )
    assert s.groq_api_key == "gsk_test"
    assert s.top_k == 15  # clamped
    assert s.candidate_k == 15  # never below top_k
    assert s.chroma_dir == str(PROJECT_ROOT / "x" / "db")
    assert s.llm_models == ("custom/model", "a", "b")  # selected model always offered
    assert s.reranker_model == ""  # empty string disables reranker


def test_secrets_used_when_env_missing():
    class BrokenSecrets(dict):
        pass

    s = load_settings(env={}, secrets={"GROQ_API_KEY": "from_secrets"})
    assert s.groq_api_key == "from_secrets"
    s2 = load_settings(env={"GROQ_API_KEY": "env"}, secrets={"GROQ_API_KEY": "secret"})
    assert s2.groq_api_key == "env"
    assert load_settings(env={}, secrets=BrokenSecrets()).groq_api_key == ""


def test_missing_secrets_file_does_not_crash():
    class Raising:
        def get(self, name):
            raise FileNotFoundError("no secrets.toml")

    assert load_settings(env={}, secrets=Raising()).groq_api_key == ""


def test_settings_are_immutable():
    s = load_settings(env={})
    with pytest.raises(AttributeError):
        s.top_k = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    "env, message",
    [
        ({"TOP_K": "abc"}, "TOP_K"),
        ({"CHROMA_MODE": "weird"}, "CHROMA_MODE"),
        ({"CHROMA_MODE": "http"}, "CHROMA_HOST"),
        ({"CHROMA_MODE": "cloud"}, "CHROMA_API_KEY"),
        ({"CHUNK_SIZE": "100"}, "CHUNK_SIZE"),
        ({"CHUNK_OVERLAP": "5000"}, "CHUNK_OVERLAP"),
        ({"EMBEDDING_BACKEND": "torchy"}, "EMBEDDING_BACKEND"),
        ({"LLM_TEMPERATURE": "hot"}, "LLM_TEMPERATURE"),
    ],
)
def test_invalid_config(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(env=env)


def test_retired_groq_models_are_replaced(caplog):
    s = load_settings(env={"GROQ_MODEL": "llama-3.1-8b-instant", "LLM_MODELS": "llama-3.3-70b-versatile"})
    assert s.llm_model == DEFAULT_LLM_MODELS[0]
    assert s.llm_models == DEFAULT_LLM_MODELS
    assert "retired" in caplog.text
