# Changelog

## 2.1.0 — 2026-11-16

### Added
- **Model failover** (`legalbot/llm.py::ModelFailover`): selected model first, then the other
  `LLM_MODELS`; switches immediately on rate limits, timeouts, connection/provider errors and
  unavailable models; never retries invalid keys, bad requests, invalid input or config errors.
- Mid-stream failures keep the partial text, mark it incomplete and offer
  "Retry with ‹next model›" without re-typing the question; history excludes interrupted answers.
- "Answered using X" notice after a fallback; per-session tracking of unavailable models.
- `legalbot/diagnostics.py` + `python -m legalbot.diagnostics`: exact fix commands for missing
  PDFs, index, Groq key, Tesseract, Bengali OCR data and embedding-model mismatches.
- Corpus integrity: mislabelled statutes are marked suspicious; `ingest.py` exits 3 unless
  `--allow-suspicious` / `ALLOW_SUSPICIOUS_DOCS=true`; the app shows a warning banner and
  flags those sources (also detected in indexes built by 2.0).
- 53 new tests (172 total), including a real `ChatGroq` + Groq SDK failover test over a mocked
  HTTP transport.

### Changed
- Questions over 2,000 characters are rejected with a clear message instead of being truncated.
- Answers without citations show a prominent "Needs verification" warning.
- `ingest.py` stops with install instructions when scanned PDFs need a missing Tesseract.
- Condense-step failures are categorised; invalid keys/config errors are no longer swallowed.

## 2.0.0 — 2026-11-15

Complete rework following the code-review roadmap.

### Fixed
- Sample questions now produce answers (previously only the question was added).
- One visitor's API key / model / Top-K no longer leaks to every other visitor
  (immutable settings; per-session state; per-request LLM).
- Model and Top-K changes in the sidebar take effect immediately (chain no longer cached once).
- Retrieval runs once per question; the sources shown are exactly what the model saw.
- Re-running ingest no longer duplicates chunks (deterministic IDs + stale cleanup).
- Default model replaced: Groq retired `llama-3.1-8b-instant` / `llama-3.3-70b-versatile`
  for free/dev tiers on 2026-08-16.
- Scanned PDFs: OCR now drops missing Tesseract language packs instead of failing every page.

### Security
- Removed all `unsafe_allow_html` rendering of user/model/document text.
- Removed `--server.enableXsrfProtection false` / CORS flags from the devcontainer.
- Per-session rate limiting; friendly error messages (no raw exceptions).
- Vector index and PDFs removed from version control.

### Added
- Section/article-aware chunking with act, section, title, chapter, part and page metadata.
- Layout-aware OCR that re-attaches margin titles; header/footer removal; OCR cache.
- Multilingual embeddings (FastEmbed, `intfloat/multilingual-e5-base`) — Bangla works.
- Hybrid retrieval (vector + BM25 + RRF), explicit section/article reference pinning,
  cross-encoder reranker.
- Citation markers `[n]` with server-side validation.
- Conversation memory via follow-up question condensing.
- **Voice**: speech input (Groq Whisper, Bangla/English), voice/text commands, read-aloud.
- Streaming answers, Chroma local/http/cloud modes, retry with backoff on Groq 429/5xx.
- `ingest.py --rebuild / --dry-run`; evaluation suite (`evals/`).
- Test suite (unit + Streamlit AppTest), ruff, strict mypy, pip-audit, Python CI matrix,
  Dockerfile with health check, docker-compose.
- Locked dependencies (`requirements.txt` via uv).

### Removed
- Node.js CI workflow (wrong language), `COMMANDS.md`, `setup.md`, `test_setup.py`
  (superseded by README and the test suite), committed vector store.
