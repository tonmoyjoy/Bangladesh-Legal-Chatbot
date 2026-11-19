# Security notes

## What the app does to protect users

* **No shared mutable state.** Settings are an immutable object; each visitor's API key,
  model and Top-K live only in their own Streamlit session. Cached resources (models,
  index) contain no secrets.
* **No raw HTML.** User text, model output and document text are rendered as Markdown with
  `unsafe_allow_html` off; source excerpts are shown as plain text. Text-to-speech payloads
  are JSON-encoded with `</` escaped.
* **XSRF protection on** (`.streamlit/config.toml`); the devcontainer no longer disables it.
* **Prompt-injection guard.** Retrieved document text is labelled as untrusted context and
  the model is told to ignore instructions inside it. Citations are validated server-side.
* **Abuse limits.** Per-session limits on questions and recordings (`RATE_LIMIT_PER_MINUTE`),
  2,000-character question limit, `MAX_AUDIO_MB` audio limit, 25 MB upload cap.
* **Errors.** Users see friendly messages; stack traces only go to the server log.
* **Logging policy.** Logs may contain model names, failure categories (e.g. `rate_limit`,
  `model_unavailable`), attempt/fallback counts, exception *class names*, file names and chunk
  counts. Logs must never contain API keys, audio, transcripts, questions, answers or retrieved
  document text. Exception messages from the provider are not logged or shown to users
  because they can echo request content. Tests assert that a key and a question placed in a
  request do not appear in the logs.
* **Model failover isolation.** Failover state (models found unavailable) is stored per
  Streamlit session; the selected model, API key and failover state of one visitor never
  affect another. Failover never changes cached, shared settings.
* **Corpus integrity.** Ingest refuses mislabelled statutes unless explicitly overridden, and
  the UI flags such documents, so users are not shown one law under another law's name.
* **Secrets.** `.env`, `.streamlit/secrets.toml`, the vector index and PDFs are git-ignored and
  excluded from the Docker image.

## Known advisories (dependency audit)

`pip-audit` reports advisories against `chromadb` (CVE-2026-45829, CVE-2026-45830,
CVE-2026-45831, CVE-2026-45833). All of them affect the **Chroma HTTP server**
(`/api/v2/...` endpoints, RBAC provider). This app uses Chroma **embedded**
(`CHROMA_MODE=local`, no server) or **Chroma Cloud**, so they do not apply. CI ignores
exactly these IDs.

If you choose `CHROMA_MODE=http` (self-hosted Chroma server): keep that server on a private
network, never expose it to the internet, enable token auth, and upgrade as soon as a fixed
release is published.

## Reporting

Please report vulnerabilities privately to the maintainer rather than opening a public issue.
